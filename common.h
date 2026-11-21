#pragma once
// ============================================================================
// common.h —— 全项目共享常量与通用小工具（唯一来源，消除各文件重复定义漂移）
// ----------------------------------------------------------------------------
// 说明：本文件为 UTF-8 with BOM，避免中文注释在 MSVC/nvcc 下乱码。
//       constexpr 常量在 host/device 代码中均可用（nvcc 12.9 验证通过）。
// ============================================================================
#include <string>
#include <cstdio>

// 圆周率（几何变换 augment/preprocessing/TTA 与 One-Cycle 余弦退火共用）。
// 以 double 精度保存，各处按原有精度使用（如 float 处显式 (float) 转换），
// 保证数值逐位不变。
static constexpr double CNN_PI = 3.14159265358979323846;

// BatchNorm 归一化分母保护（CPU bn_infer 与 GPU BN kernel 共用，均为 1e-5f）。
// 注：gradcheck 的 double 路径使用 double 字面量 1e-5（数值不同），不并入本常量。
static constexpr float CNN_BN_EPS = 1e-5f;

// 交叉熵 loss 的 log 保护（CPU 评估与 GPU kloss_acc 共用，均为 1e-8f）。
static constexpr float CNN_LOSS_EPS = 1e-8f;

// 黄金比（32 位）：种子混合 / 每 epoch 洗牌派生的固定盐值。
static constexpr unsigned CNN_GOLDEN_RATIO = 0x9E3779B9u;

// 返回 [0, n) 内最大元素的下标；比较用「严格大于」，故平局取下标最小者（首个最大）。
// 全项目原有各处 argmax 循环均为此语义，抽此助手仅为消除重复、不改变任何行为。
// 可同时用于 float/double/int 数组；对 std::vector 传 .data() 与 (int).size()。
template <typename T>
inline int argmaxIndex(const T* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

// 把秒格式化为友好字符串：<60s => "12.34 秒"，>=60s => "1 分 23.45 秒"。
// 仅依赖 <cstdio> 的 snprintf，不使用任何平台专有 API（main.cpp / gpu_trainer.cu 共用）。
inline std::string formatDur(double seconds) {
    char buf[64];
    if (seconds < 60.0) {
        std::snprintf(buf, sizeof(buf), "%.2f 秒", seconds);
    } else {
        int totalSec = (int)seconds;
        int m = totalSec / 60;
        double s = seconds - m * 60.0;
        std::snprintf(buf, sizeof(buf), "%d 分 %.2f 秒", m, s);
    }
    return std::string(buf);
}
