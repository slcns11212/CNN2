#pragma once
// ============================================================================
// preprocessing.h —— 可配置的输入预处理模块（header-only）
// ----------------------------------------------------------------------------
// 目的：把两套来源不同的手写数字数据（optdigits / MNIST）做几何归一化对齐。
// 核心：严格保持二值（0/1）。所有几何变换只用整数平移 / 最近邻 / 覆盖率阈值，
//       不做任何灰度插值，因此不会破坏二值约束。
// 注意：本文件为 UTF-8 with BOM，避免中文注释在 nvcc/MSVC 下乱码。
// ============================================================================

#include "conv.h"
#include "common.h"     // 共享常量（CNN_PI）
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

// ============================================================================
// 配置结构
// ============================================================================
struct PrepConfig {
    // 各步骤开关（默认启用 crop, resize, center；关闭 deskew, thick）
    bool  doCrop = true;
    bool  doDeskew = false;
    bool  doResize = true;
    bool  doCenter = true;
    bool  doThick = false;

    // resize 参数
    int   target = 28;      // 目标框最长边（建议 16-30）
    bool  stretch = false;  // true：拉伸到 target×target；false：保持长宽比

    // thicknessNormalize 参数
    float targetInk = 0.20f; // 目标墨迹比（0 表示关闭）
    int   thickMode = 0;     // 0=自动（按当前墨迹比选择膨胀/腐蚀），1=只膨胀，2=只腐蚀
    int   thickIters = 5;    // 最大迭代次数
    float thickTol = 0.01f;  // 收敛容差

    // deskew 参数
    float deskewMaxDeg = 15.0f; // 角度限幅（度）

    // 诊断
    bool  diag = false;     // 打印预处理前后统计
};

// ============================================================================
// 统计信息
// ============================================================================
struct PrepStats {
    float inkRatio = 0.0f;   // 墨迹比（前景像素 / 全部像素）
    float cx = 0.0f;         // 质心 x（列，0 基）
    float cy = 0.0f;         // 质心 y（行，0 基）
    int   minH = 0, maxH = -1; // 包围盒行范围（无前景时 maxH<minH）
    int   minW = 0, maxW = -1; // 包围盒列范围
    int   inkCount = 0;      // 前景像素数
    bool  hasForeground = false;
};

// 判定前景的阈值（二值图里恒为 1）
inline bool prepIsForeground(float v) { return v > 0.5f; }

// ============================================================================
// computeStats：返回墨迹比、质心、包围盒
// ============================================================================
inline PrepStats computeStats(const ConvMap& img) {
    PrepStats s;
    s.minH = img.H; s.maxH = -1;
    s.minW = img.W; s.maxW = -1;

    double sx = 0.0, sy = 0.0;
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                ++s.inkCount;
                sx += w; sy += h;
                if (h < s.minH) s.minH = h;
                if (h > s.maxH) s.maxH = h;
                if (w < s.minW) s.minW = w;
                if (w > s.maxW) s.maxW = w;
            }

    int total = img.C * img.H * img.W;
    s.inkRatio = (total > 0) ? (float)s.inkCount / (float)total : 0.0f;
    s.hasForeground = (s.inkCount > 0);
    if (s.hasForeground) {
        s.cx = (float)(sx / s.inkCount);
        s.cy = (float)(sy / s.inkCount);
    }
    else {
        s.cx = img.W * 0.5f;
        s.cy = img.H * 0.5f;
    }
    return s;
}

// ============================================================================
// isBinary：诊断断言，检查是否严格为 0/1
// ============================================================================
inline bool isBinary(const ConvMap& img) {
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                float v = img.data[c][h][w];
                if (v != 0.0f && v != 1.0f) return false;
            }
    return true;
}

// 是否存在前景
inline bool hasForeground(const ConvMap& img) {
    return computeStats(img).hasForeground;
}

// 把任意非零灰度强制归到 0/1（保证总入口输出二值）
inline void finalizeBinary(ConvMap& img) {
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w)
                img.data[c][h][w] = prepIsForeground(img.data[c][h][w]) ? 1.0f : 0.0f;
}

// ============================================================================
// 内部工具：按整数 (dx,dy) 平移，越界像素被裁剪丢弃
// ============================================================================
inline void prepShift(ConvMap& img, int dx, int dy) {
    if (dx == 0 && dy == 0) return;
    ConvMap out(img.C, img.H, img.W);
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                int nh = h + dy;
                int nw = w + dx;
                if (nh < 0 || nh >= img.H) continue; // 裁剪保护
                if (nw < 0 || nw >= img.W) continue;
                out.data[c][nh][nw] = 1.0f;
            }
    img = out;
}

// ============================================================================
// cropToBBox：扫描前景包围盒，平移到 (0,0)；无前景返回 false
// ============================================================================
inline bool cropToBBox(ConvMap& img) {
    PrepStats s = computeStats(img);
    if (!s.hasForeground) return false;
    prepShift(img, -s.minW, -s.minH);
    return true;
}

// ============================================================================
// resizeToTarget：内容缩放到目标框（默认保持长宽比）
//   - 下采样：覆盖率阈值 t = 0.5
//   - 上采样：最近邻
//   - 严格保持二值
// ============================================================================
inline void resizeToTarget(ConvMap& img, int target, bool stretch) {
    PrepStats s = computeStats(img);
    if (!s.hasForeground) return;

    int bboxH = s.maxH - s.minH + 1;
    int bboxW = s.maxW - s.minW + 1;
    if (bboxH <= 0 || bboxW <= 0) return;

    int newH, newW;
    if (stretch) {
        newH = target;
        newW = target;
    }
    else {
        // 保持长宽比：最长边缩放到 target
        float scale = (float)target / (float)(std::max)(bboxH, bboxW);
        newH = (std::max)(1, (int)std::lround(bboxH * scale));
        newW = (std::max)(1, (int)std::lround(bboxW * scale));
    }
    newH = (std::min)(newH, img.H);
    newW = (std::min)(newW, img.W);

    ConvMap out(img.C, img.H, img.W);
    int oy = (img.H - newH) / 2; // 内容先居中放置，后续 centerByCentroid 会精确对齐质心
    int ox = (img.W - newW) / 2;

    const float t = 0.5f; // 覆盖率阈值

    for (int c = 0; c < img.C; ++c) {
        for (int oh = 0; oh < newH; ++oh) {
            // 输出行 oh 对应的源包围盒行区间 [sy0, sy1)
            float sy0 = (float)oh * bboxH / newH;
            float sy1 = (float)(oh + 1) * bboxH / newH;
            int r0 = (int)std::floor(sy0);
            int r1 = (int)std::floor(sy1 - 1e-6f);
            if (r1 < r0) r1 = r0;

            for (int ow = 0; ow < newW; ++ow) {
                float sx0 = (float)ow * bboxW / newW;
                float sx1 = (float)(ow + 1) * bboxW / newW;
                int c0 = (int)std::floor(sx0);
                int c1 = (int)std::floor(sx1 - 1e-6f);
                if (c1 < c0) c1 = c0;

                int fg = 0, cnt = 0;
                for (int r = r0; r <= r1; ++r) {
                    int sr = s.minH + r;
                    if (sr < 0 || sr >= img.H) continue;
                    for (int cc = c0; cc <= c1; ++cc) {
                        int sc = s.minW + cc;
                        if (sc < 0 || sc >= img.W) continue;
                        ++cnt;
                        if (prepIsForeground(img.data[c][sr][sc])) ++fg;
                    }
                }
                // 覆盖率达到阈值才置 1（下采样面积平均；上采样自然退化为最近邻）
                if (cnt > 0 && (float)fg / (float)cnt >= t)
                    out.data[c][oy + oh][ox + ow] = 1.0f;
            }
        }
    }
    img = out;
}

// ============================================================================
// centerByCentroid：前景质心整数平移到画布中心 (15.5,15.5)（对 32×32）
// ============================================================================
inline void centerByCentroid(ConvMap& img) {
    PrepStats s = computeStats(img);
    if (!s.hasForeground) return;

    int dx = (int)std::lround((img.W - 1) * 0.5 - s.cx);
    int dy = (int)std::lround((img.H - 1) * 0.5 - s.cy);
    if (dx == 0 && dy == 0) return;
    prepShift(img, dx, dy);
}

// ============================================================================
// deskewImage：基于二阶矩的剪切校正（仅水平剪切）
//   s = μ11 / μ02，水平剪切 x' = x + s(y-yc)
//   - 角度限幅到 ±maxDeg
//   - 剪切前质心居中，防裁切；剪切后再居中
//   - 低各向异性（接近圆）则跳过
//   - 最近邻采样，严格二值
//   - E8：原 axis 参数在唯一调用点恒为 0，垂直剪切分支不可达，已移除
// ============================================================================
inline void deskewImage(ConvMap& img, float maxDeg) {
    PrepStats s = computeStats(img);
    if (!s.hasForeground || s.inkCount < 2) return;

    double mu11 = 0.0, mu02 = 0.0, mu20 = 0.0;
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                double dx = w - s.cx;
                double dy = h - s.cy;
                mu11 += dx * dy;
                mu02 += dy * dy;
                mu20 += dx * dx;
            }
    double n = (double)s.inkCount;
    mu11 /= n; mu02 /= n; mu20 /= n;

    double denom = mu02;
    if (denom < 1e-3) return; // 退化（近似水平/垂直直线）

    // 低各向异性跳过（接近圆，方向不可靠）
    double aniso = std::fabs(mu20 - mu02) / (mu20 + mu02 + 1e-6);
    if (aniso < 0.02) return;

    double slope = mu11 / denom;
    if (std::fabs(slope) < 1e-4) return;

    // 角度限幅
    double maxRad = (double)maxDeg * CNN_PI / 180.0;
    double ang = std::atan(slope);
    if (ang > maxRad) ang = maxRad;
    if (ang < -maxRad) ang = -maxRad;
    slope = std::tan(ang);

    // 剪切前质心居中，防止裁切
    centerByCentroid(img);
    s = computeStats(img);

    ConvMap out(img.C, img.H, img.W);
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                // 逆映射：正向 x' = x + s(y-yc) → x = x' - s(y-yc)
                int sw = (int)std::lround((double)w - slope * ((double)h - s.cy));
                if (sw < 0 || sw >= img.W) continue;
                out.data[c][h][sw] = 1.0f;
            }
    img = out;

    // 剪切后再居中，防裁切
    centerByCentroid(img);
}

// ============================================================================
// 3×3 形态学膨胀 / 腐蚀（8 邻域，二值）
// ============================================================================
inline void dilate3x3(ConvMap& img) {
    ConvMap out(img.C, img.H, img.W);
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                for (int dh = -1; dh <= 1; ++dh) {
                    int nh = h + dh;
                    if (nh < 0 || nh >= img.H) continue;
                    for (int dw = -1; dw <= 1; ++dw) {
                        int nw = w + dw;
                        if (nw < 0 || nw >= img.W) continue;
                        out.data[c][nh][nw] = 1.0f;
                    }
                }
            }
    img = out;
}

inline void erode3x3(ConvMap& img) {
    ConvMap out(img.C, img.H, img.W);
    for (int c = 0; c < img.C; ++c)
        for (int h = 0; h < img.H; ++h)
            for (int w = 0; w < img.W; ++w) {
                if (!prepIsForeground(img.data[c][h][w])) continue;
                bool all = true;
                for (int dh = -1; dh <= 1 && all; ++dh) {
                    int nh = h + dh;
                    if (nh < 0 || nh >= img.H) { all = false; break; }
                    for (int dw = -1; dw <= 1; ++dw) {
                        int nw = w + dw;
                        if (nw < 0 || nw >= img.W) { all = false; break; }
                        if (!prepIsForeground(img.data[c][nh][nw])) { all = false; break; }
                    }
                }
                if (all) out.data[c][h][w] = 1.0f;
            }
    img = out;
}

// ============================================================================
// thicknessNormalize：按目标墨迹比迭代 3×3 膨胀/腐蚀
//   mode: 0=自动（依据当前墨迹比选择），1=只膨胀，2=只腐蚀
//   腐蚀后为空则回滚
//   E8：mode 目前恒为 0——prepConfigFromArgs 从不解析 thickMode（保持默认 0=自动），
//       故 mode==1/2 分支当前不可达。此处选择「保留 + 注释」而非删除：该参数属
//       PrepConfig 的公开配置项，保留可避免改动配置结构，便于将来接入 CLI 开关；
//       README 亦明确记载「内部固定 thickMode=0（自动）」。
// ============================================================================
inline void thicknessNormalize(ConvMap& img, float targetInk, int mode,
    int iters, float tol) {
    if (targetInk <= 0.0f) return;
    if (iters < 1) iters = 1;

    for (int it = 0; it < iters; ++it) {
        float ink = computeStats(img).inkRatio;
        if (std::fabs(ink - targetInk) <= tol) break;

        bool doDilate;
        if (mode == 1) doDilate = true;
        else if (mode == 2) doDilate = false;
        else doDilate = (ink < targetInk);

        if (doDilate) {
            dilate3x3(img);
        }
        else {
            ConvMap prev = img; // 回滚点
            erode3x3(img);
            if (!hasForeground(img)) { img = prev; break; }
        }
    }
}

// ============================================================================
// preprocessImage：预处理管线总入口
//   默认顺序：cropToBBox → deskew → resizeToTarget → centerByCentroid → thicknessNormalize
//   保证输出仍为 32×32 二值
// ============================================================================
inline void preprocessImage(ConvMap& img, const PrepConfig& cfg) {
    if (cfg.doCrop)   cropToBBox(img);
    if (cfg.doDeskew) deskewImage(img, cfg.deskewMaxDeg);
    if (cfg.doResize) resizeToTarget(img, cfg.target, cfg.stretch);
    if (cfg.doCenter) centerByCentroid(img);
    if (cfg.doThick)  thicknessNormalize(img, cfg.targetInk, cfg.thickMode,
        cfg.thickIters, cfg.thickTol);
    finalizeBinary(img); // 总入口保证二值
}

// ============================================================================
// prepConfigFromArgs：从命令行参数 map 装配配置
//   --prep crop,resize,center | none
//   --prep_target N (16-30)
//   --prep_stretch
//   --prep_ink X
//   --prep_deskew_max D
//   --diag
// ============================================================================
inline PrepConfig prepConfigFromArgs(const std::map<std::string, std::string>& args) {
    PrepConfig cfg;

    // ---- --prep 步骤清单 ----
    std::map<std::string, std::string>::const_iterator it = args.find("prep");
    if (it != args.end()) {
        // 显式指定了 --prep：先全部关闭，再按清单开启
        cfg.doCrop = cfg.doDeskew = cfg.doResize = cfg.doCenter = cfg.doThick = false;

        std::string v = it->second;
        if (v != "none" && v != "1") {
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                // 去首尾空白
                size_t b = tok.find_first_not_of(" \t");
                size_t e = tok.find_last_not_of(" \t");
                if (b == std::string::npos) continue;
                tok = tok.substr(b, e - b + 1);
                if (tok == "crop")        cfg.doCrop = true;
                else if (tok == "deskew") cfg.doDeskew = true;
                else if (tok == "resize") cfg.doResize = true;
                else if (tok == "center") cfg.doCenter = true;
                else if (tok == "thick")  cfg.doThick = true;
            }
        }
    }

    // ---- 数值参数 ----
    it = args.find("prep_target");
    if (it != args.end()) {
        int t = std::atoi(it->second.c_str());
        if (t < 16) t = 16;
        if (t > 30) t = 30;
        cfg.target = t;
    }

    if (args.count("prep_stretch")) cfg.stretch = true;

    it = args.find("prep_ink");
    if (it != args.end()) cfg.targetInk = (float)std::atof(it->second.c_str());

    it = args.find("prep_deskew_max");
    if (it != args.end()) cfg.deskewMaxDeg = (float)std::atof(it->second.c_str());

    if (args.count("diag")) cfg.diag = true;

    return cfg;
}
