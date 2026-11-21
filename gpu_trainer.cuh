// gpu_trainer.cuh
// GPU 训练模块对外接口（纯 C++，不引入 CUDA 头文件，供 main.cpp 包含）。
// 通过「扁平参数 + epoch 回调」与主程序解耦，避免依赖 conv.h / sf.h 中的
// 非 inline 全局定义（那些定义已在 main.cpp 所在编译单元中存在）。
#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "net_spec.h"   // 运行时网络规格：主机与 GPU 用同一份 NetSpec
#include "common.h"     // 共享常量（CNN_PI / CNN_GOLDEN_RATIO）

// ================= 阶段4.1：确定性种子派生（CPU/GPU 共用） =================
// 由 (seed, salt) 混合出独立的子种子，供初始化 / 每 epoch 洗牌 / 增强各自使用，
// 避免不同随机源相互耦合。纯整数运算、无平台依赖，main.cpp 与 gpu_trainer.cu
// 共用同一实现，保证「同一 seed → 同一序列」。
inline unsigned mixSeed(unsigned seed, unsigned salt) {
    unsigned x = seed ^ (salt * CNN_GOLDEN_RATIO);
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

// ================= One-Cycle 学习率调度（GPU/CPU 共用） =================
// 默认 oneCycle=false：lrScaleFactor 恒返回 1.0f，训练完全等价于固定 LR。
// 开启后：前 warmupSteps 从 base*minFactor 线性升到 base*peakFactor，
//         其后余弦退火回到 base*minFactor（step 为 per-epoch 或 per-batch 粒度）。
struct LrScheduleConfig {
    bool  oneCycle     = false;
    int   warmupEpochs = 0;      // 0 = auto (max(1, epochs/5))
    float minFactor    = 0.01f;  // 最小 lr = base * minFactor
    float peakFactor   = 1.0f;   // 峰值 lr = base * peakFactor
};

// 返回施加到 base lr 上的倍率：非 oneCycle 恒为 1.0f（保证默认逐位一致）。
inline float lrScaleFactor(const LrScheduleConfig& s, long long step,
                           long long totalSteps, long long warmupSteps) {
    if (!s.oneCycle) return 1.0f;
    float pk = s.peakFactor, mn = s.minFactor;
    if (mn < 0) mn = 0; if (mn > pk) mn = pk;
    if (totalSteps <= 1) return pk;
    if (warmupSteps < 0) warmupSteps = 0;
    if (warmupSteps > totalSteps) warmupSteps = totalSteps;
    if (step < 0) step = 0;
    if (step >= totalSteps) step = totalSteps - 1;
    if (step < warmupSteps) {
        float u = warmupSteps > 0 ? (float)step / (float)warmupSteps : 1.0f;
        return mn + (pk - mn) * u;
    }
    long long d = totalSteps - warmupSteps;
    float p = d > 0 ? (float)(step - warmupSteps) / (float)d : 1.0f;
    const float PI = (float)CNN_PI;   // 保持 float 精度与原字面量 3.14159265358979f 逐位一致
    return mn + 0.5f * (pk - mn) * (1.0f + cosf(PI * p));
}

struct GpuTrainConfig {
    int   epochs    = 10;
    int   batch     = 128;
    float lr_fc     = 0.0001f;
    float lr_conv1  = 0.0001f;
    float lr_conv2  = 0.0005f;
    bool  train_conv1 = true;
    bool  train_conv2 = true;
    bool  train_fc    = true;
    float momentum  = 0.9f;    // SGD momentum 系数 β（0 = 关闭，退化为朴素 SGD）
    bool  nesterov  = false;   // 是否启用 Nesterov look-ahead
    float weightDecay = 0.0f;  // L2 weight decay 系数（--wd，默认 0.0=关闭，仅乘法权重）
    bool  timeIt    = false;   // 是否输出每个 epoch 的耗时（--time，跨平台 steady_clock）
    unsigned seed   = 0;       // 阶段4.1：训练随机种子（--seed）；用于每 epoch 确定性洗牌
    LrScheduleConfig lrSched;  // One-Cycle 学习率调度（默认关闭，固定 LR）
};

// 每个 epoch 结束回调：params 为 host 端扁平参数 [spec.total]。
// 主程序可借此把参数同步回全局权重并复用 saveModel() 落盘。
// 阶段4.3（可观测性）：同时携带本 epoch 统计量，便于主程序写训练日志（--log）：
//   loss/acc      : 本 epoch 平均 loss 与准确率（与 stdout 打印同源）
//   lr_conv1/2/fc : 本 epoch 末实际生效学习率（one-cycle 为退火后值；默认即 base lr）
//   epochSec      : 本 epoch 墙上耗时（秒，steady_clock）
typedef void (*GpuEpochCallback)(const float* params, int epoch,
                                 float loss, float acc,
                                 float lr_conv1, float lr_conv2, float lr_fc,
                                 double epochSec, void* userData);

// 是否存在可用 CUDA 设备（不抛异常，任何错误都返回 false）。
bool gpuAvailable();

// 获取当前 CUDA 设备名称（cudaDeviceProp.name）。
// 无可用设备时返回 "CPU"；查询失败返回 "N/A"。纯 host 接口，供 main.cpp 调用。
std::string gpuName();

// 执行全 GPU 训练（前向 + 反向）。
//  spec   : 运行时网络规格（维度/偏移的唯一来源，须经 validateNetSpec 校验）
//  data   : 连续输入，大小 = N * spec.inH*spec.inW，元素取值 0/1
//  labels : 大小 = N，取值范围 [0, spec.classes)
//  cfg    : 训练超参
//  params : in = 初始参数[spec.total]；out = 训练结束后参数[spec.total]
//  onEpochEnd : 可为 nullptr
// 返回 true 表示成功。
bool gpuTrain(const std::vector<float>& data,
              const std::vector<int>& labels,
              const GpuTrainConfig& cfg,
              std::vector<float>& params,
              GpuEpochCallback onEpochEnd,
              void* userData,
              const NetSpec& spec);

// 批量前向推理（test --gpu 用）：只跑前向（Conv/BN/ReLU/Pool 与 FC+Softmax），
// 不建图不求梯度；BN 模型按 eval 模式用 running 统计量（不更新 running）。
// 支持多模型概率平均集成：对 paramsList 中每个模型各跑一遍前向，把每样本的
// softmax 概率逐类累加，最后取平均概率的 argmax（M=1 时与单模型结果一致）。
//  data      : 连续输入，大小 = N * spec.inH*spec.inW，元素 0/1
//  paramsList: 每个元素为该模型的扁平参数 [spec.total]，不可为空
//  batch     : 批大小；<=0 或 > N 时按 N 处理
//  outPred   : 输出，长度 N，元素为平均概率的 argmax 预测标签
//  outAllProb: 可选输出；非 nullptr 时填充 N*M*spec.classes 的逐模型概率
//              （M = paramsList.size()），索引
//              ((size_t)m * N + i) * spec.classes + c 表示第 m 个模型对
//              第 i 个样本第 c 类的 softmax 概率；nullptr 时零开销跳过
// 返回 true 表示成功。
bool gpuPredict(const std::vector<float>& data,
                int N,
                const std::vector<std::vector<float>>& paramsList,
                int batch,
                std::vector<int>& outPred,
                std::vector<float>* outAllProb = nullptr,
                const NetSpec& spec = NetSpec{});   // 默认参数置于末尾，避免破坏既有调用
