#pragma once
// ============================================================================
// augment.h —— 数据增强模块（header-only）
// ----------------------------------------------------------------------------
// 目的：对已预处理的 32×32 二值手写数字做随机增强，提升跨域泛化。
// 原则：严格保持二值（0/1）。所有几何变换用「逆向映射 + 最近邻」，
//       弹性形变用「随机位移场 + 高斯平滑 + 最近邻重采样 + 重新二值化」
//       （Simard 2003 方法）。不做任何灰度插值，输出仍严格为 0/1。
// 注意：本文件为 UTF-8 with BOM，避免中文注释在 MSVC/nvcc 下乱码。
// ============================================================================

#include "conv.h"
#include "preprocessing.h"
#include "common.h"     // 共享常量（CNN_PI）

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>
#include <random>

// ============================================================================
// 增强配置
// ============================================================================
struct AugConfig {
    bool  enable = false;       // --aug 总开关（默认关，保持向后兼容）

    int   copies = 2;           // --aug_k：GPU 路径离线扩增副本数（每张原图额外生成 K 张）
    int   shift = 3;            // --aug_shift：随机平移幅度 ±px
    float rotateDeg = 12.0f;    // --aug_rotate：随机旋转 ±度
    float scaleJitter = 0.1f;   // --aug_scale：缩放抖动半幅（0.1 => 0.9~1.1）
    float elasticAlpha = 0.0f;  // --aug_elastic：弹性形变强度 α（0=关闭，建议从 4 起）
    float elasticSigma = 4.5f;  // --aug_elastic_sigma：位移场高斯平滑 σ（建议 4~5）
    int   thickPct = 50;        // --aug_thick：笔画粗细增强触发概率（%），0=关闭
};

// 从命令行参数 map 装配增强配置（参数名不带前导 --）
inline AugConfig augConfigFromArgs(const std::map<std::string, std::string>& args) {
    AugConfig c;

    if (args.count("aug")) c.enable = true;

    std::map<std::string, std::string>::const_iterator it;
    it = args.find("aug_k");
    if (it != args.end()) { int k = std::atoi(it->second.c_str()); if (k < 0) k = 0; c.copies = k; }
    it = args.find("aug_shift");
    if (it != args.end()) c.shift = std::atoi(it->second.c_str());
    it = args.find("aug_rotate");
    if (it != args.end()) c.rotateDeg = (float)std::atof(it->second.c_str());
    it = args.find("aug_scale");
    if (it != args.end()) c.scaleJitter = (float)std::atof(it->second.c_str());
    it = args.find("aug_elastic");
    if (it != args.end()) c.elasticAlpha = (float)std::atof(it->second.c_str());
    it = args.find("aug_elastic_sigma");
    if (it != args.end()) c.elasticSigma = (float)std::atof(it->second.c_str());
    it = args.find("aug_thick");
    if (it != args.end()) c.thickPct = std::atoi(it->second.c_str());

    if (c.shift < 0) c.shift = 0;
    if (c.rotateDeg < 0.0f) c.rotateDeg = -c.rotateDeg;
    if (c.scaleJitter < 0.0f) c.scaleJitter = -c.scaleJitter;
    // M4：弹性形变参数上限防护。sigma 直接决定 gaussianSmoothField 的
    // radius=ceil(3σ) 与核/临时缓冲大小，过大会 bad_alloc，故 clamp 到 [0.5,50]；
    // alpha 为位移幅度，clamp 到 [0,50] 防止采样失控。默认值在区间内，行为不变。
    if (c.elasticSigma < 0.5f) c.elasticSigma = 0.5f;
    if (c.elasticSigma > 50.0f) c.elasticSigma = 50.0f;
    if (c.elasticAlpha < 0.0f) c.elasticAlpha = 0.0f;
    if (c.elasticAlpha > 50.0f) c.elasticAlpha = 50.0f;
    if (c.thickPct < 0) c.thickPct = 0;
    if (c.thickPct > 100) c.thickPct = 100;
    return c;
}

// ============================================================================
// 几何增强：随机平移 + 旋转 + 缩放
//   采用逆向映射 + 最近邻：对每个输出像素反算源坐标，越界丢弃，
//   源为前景则置 1。因此输出严格二值。
// ============================================================================
inline void augmentGeometric(ConvMap& img, int maxShift, float maxRotDeg,
    float scaleJitter, std::mt19937& rng)
{
    if (maxShift <= 0 && maxRotDeg <= 0.0f && scaleJitter <= 0.0f) return;

    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ui(-maxShift, maxShift);

    int   dx = (maxShift > 0) ? ui(rng) : 0;
    int   dy = (maxShift > 0) ? ui(rng) : 0;
    float deg = (maxRotDeg > 0.0f) ? ur(rng) * maxRotDeg : 0.0f;
    float sc = 1.0f + ((scaleJitter > 0.0f) ? ur(rng) * scaleJitter : 0.0f);
    if (sc < 0.5f) sc = 0.5f;
    if (sc > 2.0f) sc = 2.0f;

    if (dx == 0 && dy == 0 && deg == 0.0f && sc == 1.0f) return;

    double rad = (double)deg * CNN_PI / 180.0;
    float cosT = (float)std::cos(rad);
    float sinT = (float)std::sin(rad);
    float cx = (img.W - 1) * 0.5f;
    float cy = (img.H - 1) * 0.5f;
    float inv = 1.0f / sc; // 逆向缩放

    ConvMap out(img.C, img.H, img.W);
    for (int oh = 0; oh < img.H; ++oh) {
        for (int ow = 0; ow < img.W; ++ow) {
            // 逆变换：out 坐标 -> 源坐标（去平移、去旋转、去缩放）
            float x = (float)ow - cx - dx;
            float y = (float)oh - cy - dy;
            float xs = (cosT * x + sinT * y) * inv + cx;
            float ys = (-sinT * x + cosT * y) * inv + cy;
            int sx = (int)std::lround(xs);
            int sy = (int)std::lround(ys);
            if (sx < 0 || sx >= img.W || sy < 0 || sy >= img.H) continue;
            for (int c = 0; c < img.C; ++c)
                if (prepIsForeground(img.data[c][sy][sx]))
                    out.data[c][oh][ow] = 1.0f;
        }
    }
    img = out;
    finalizeBinary(img);
}

// ============================================================================
// 可分离高斯平滑（用于弹性形变的随机位移场）
//   边界用镜像反射，radius 超过画布时退化为钳制。
// ============================================================================
inline void gaussianSmoothField(std::vector<float>& f, int H, int W, float sigma) {
    if (sigma <= 0.0f || H <= 0 || W <= 0) return;
    int radius = (int)std::ceil(3.0f * sigma);
    if (radius < 1) radius = 1;

    std::vector<float> k(2 * radius + 1);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
        k[i + radius] = v;
        sum += v;
    }
    for (int i = 0; i < (int)k.size(); ++i) k[i] /= sum;

    // 横向卷积
    std::vector<float> tmp((size_t)H * W);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                int ww = w + i;
                if (ww < 0) ww = -ww;
                if (ww >= W) ww = 2 * W - 2 - ww;
                if (ww < 0) ww = 0;
                if (ww >= W) ww = W - 1;
                acc += k[i + radius] * f[(size_t)h * W + ww];
            }
            tmp[(size_t)h * W + w] = acc;
        }
    }
    // 纵向卷积
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                int hh = h + i;
                if (hh < 0) hh = -hh;
                if (hh >= H) hh = 2 * H - 2 - hh;
                if (hh < 0) hh = 0;
                if (hh >= H) hh = H - 1;
                acc += k[i + radius] * tmp[(size_t)hh * W + w];
            }
            f[(size_t)h * W + w] = acc;
        }
    }
}

// ============================================================================
// 弹性形变 elasticDeform（Simard 2003）
//   1) 生成两张 [-1,1] 均匀随机位移场 dx/dy
//   2) 分别用 σ 的高斯核平滑
//   3) 输出像素 (h,w) 采样源 (h+α·dy, w+α·dx)，最近邻
//   4) 重新二值化（finalizeBinary）
// ============================================================================
inline void augmentElastic(ConvMap& img, float alpha, float sigma, std::mt19937& rng) {
    if (alpha <= 0.0f) return;

    // M4 使用处兜底：即使调用方绕过 augConfigFromArgs 直接构造参数，也把
    // sigma/alpha 的上限钳制到安全区间，避免 gaussianSmoothField 的
    // radius=ceil(3σ) 过大导致 bad_alloc。默认值在区间内，行为不变。
    if (sigma > 50.0f) sigma = 50.0f;
    if (alpha > 50.0f) alpha = 50.0f;

    int H = img.H, W = img.W;
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);

    std::vector<float> dxf((size_t)H * W), dyf((size_t)H * W);
    // L4：循环上界用 (size_t)H*W，避免 H*W 以 int 相乘时溢出
    for (size_t i = 0; i < (size_t)H * W; ++i) {
        dxf[i] = ur(rng);
        dyf[i] = ur(rng);
    }
    gaussianSmoothField(dxf, H, W, sigma);
    gaussianSmoothField(dyf, H, W, sigma);

    ConvMap out(img.C, H, W);
    for (int oh = 0; oh < H; ++oh) {
        for (int ow = 0; ow < W; ++ow) {
            size_t idx = (size_t)oh * W + ow;
            float sx = (float)ow + alpha * dxf[idx];
            float sy = (float)oh + alpha * dyf[idx];
            int ix = (int)std::lround(sx);
            int iy = (int)std::lround(sy);
            if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
            for (int c = 0; c < img.C; ++c)
                if (prepIsForeground(img.data[c][iy][ix]))
                    out.data[c][oh][ow] = 1.0f;
        }
    }
    img = out;
    finalizeBinary(img);
}

// ============================================================================
// 笔画粗细增强：随机 3×3 膨胀/腐蚀（模拟不同采集方式的粗细差）
//   pct 为触发概率（%）。腐蚀后若为空则回滚。
// ============================================================================
inline void augmentThickness(ConvMap& img, int pct, std::mt19937& rng) {
    if (pct <= 0) return;
    std::uniform_int_distribution<int> ui(1, 100);
    if (ui(rng) > pct) return;

    std::uniform_int_distribution<int> coin(0, 1);
    if (coin(rng) == 0) {
        dilate3x3(img);
    }
    else {
        ConvMap prev = img; // 回滚点
        erode3x3(img);
        if (!hasForeground(img)) img = prev;
    }
    finalizeBinary(img);
}

// ============================================================================
// 单样本随机增强总入口
//   顺序：几何 -> 弹性 -> 笔画粗细 -> 二值化
// ============================================================================
inline void augmentRandom(ConvMap& img, const AugConfig& cfg, std::mt19937& rng) {
    augmentGeometric(img, cfg.shift, cfg.rotateDeg, cfg.scaleJitter, rng);
    if (cfg.elasticAlpha > 0.0f)
        augmentElastic(img, cfg.elasticAlpha, cfg.elasticSigma, rng);
    augmentThickness(img, cfg.thickPct, rng);
    finalizeBinary(img);
}

