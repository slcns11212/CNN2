// gpu_kernels.cuh
// GPU 训练模块：网络维度、参数扁平布局、显存缓冲结构、CUDA kernel 声明。
// 注意：本头文件只被 .cu 编译单元包含，main.cpp 不包含它。
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "net_spec.h"   // 运行时网络规格（与主机侧共用同一份定义）
#include "common.h"     // 共享常量（CNN_BN_EPS）

// ================= 网络维度 =================
// 阶段2 起 GPU 由运行时 NetSpec 驱动：维度/偏移一律取自 kernel 传入的 NetSpec N，
// 不再使用编译期默认规模宏。
// H3：MAX_CLASSES 的唯一定义已移至 net_spec.h（static constexpr int，值 16）。
// 此处不再重复定义，只做编译期下限校验；kb_fc 的 __shared__ dlog[MAX_CLASSES] 依赖该上限。
// 运行期 classes 的上界由 validateNetSpec（net_spec.h）拦截（classes ∈ [2,16]）。
static_assert(MAX_CLASSES >= 16, "MAX_CLASSES must be >= 16 (kb_fc shared dlog)");

// ================= 参数扁平布局 =================
// 参数偏移/数量由 makeDefaultNetSpec（net_spec.h）在运行时计算并随 NetSpec 传入，
// 此处不再定义默认规模宏；维度自洽由 validateNetSpec（net_spec.h）统一保证。

#define CNN_BLOCK  256

// L2：kbn_stats / kbn_bwd / kloss_acc 的树形归约（stride = blockDim>>1）隐含
// 假设 blockDim 为 2 的幂。对 CNN_BLOCK 在编译期强制该前提，防止将来改错。
static_assert((CNN_BLOCK & (CNN_BLOCK - 1)) == 0,
              "CNN_BLOCK must be a power of two for tree reductions");

// ================= BN 常量（CPU/GPU/文件三处一致，P2） =================
// eps        : 归一化分母保护（唯一定义见 common.h 的 CNN_BN_EPS，CPU bn_infer 同源）
// BN_MOMENTUM: running = (1-m)*running + m*batch
// 归一化统计量用「有偏方差」(÷N)；写 running var 时用「无偏」(×N/(N-1))。
#define BN_EPS      CNN_BN_EPS
#define BN_MOMENTUM 0.1f

// ================= 显存缓冲结构 =================
// 除 d_data / d_allLabel / d_index / d_params / d_grad 为全局常驻外，
// 其余均为「按 batch 大小分配」的批内缓冲（每个样本一段，块间私有）。
// 大小均按传入的 NetSpec 计算（不再用编译期宏）。
struct GpuBuffers {
    float* d_data;      // [N * inH*inW]        全量输入（常驻）
    int*   d_allLabel;  // [N]                  全量标签（常驻）
    int*   d_index;     // [N]                  当前 epoch 的乱序索引（常驻）
    float* d_input;     // [batch*inH*inW]      当前 batch 输入（gather 后）
    int*   d_label;     // [batch]              当前 batch 标签
    // ===== 逐层缓冲（数组池，下标 li = 0-based 层号） =====
    // d_c[li]   : 该层 conv 原始输出（BN/ReLU 前）
    // d_r[li]   : 该层 ReLU 输出
    // d_p[li]   : 该层 Pool 输出（可作为下一层输入；末层即 flatten）
    float* d_c[MAX_DEPTH];
    float* d_r[MAX_DEPTH];
    float* d_p[MAX_DEPTH];
    float* d_logits;    // [batch*classes]
    float* d_prob;      // [batch*classes]
    float* d_dflat;     // [batch*flat]         末层 Pool 输出梯度，样本私有
    float* d_dp[MAX_DEPTH];  // 该层 Pool 输出梯度，样本私有
    float* d_dc[MAX_DEPTH];  // 该层 conv 梯度，样本私有
    // ===== 可选 BN（仅 spec.bn==true 时分配，否则保持 nullptr） =====
    // d_b[li]  : BN 输出（ReLU 前）。非 BN 路径完全不用。
    // d_db[li] : BN 输出梯度（回传到 conv 前的中间梯度）。
    // d_bn_mean/var[li]: 前向算出的「本 batch」统计量（有偏方差），反向复用。
    // 训练时 running 统计量本身存放在 d_params 的 offBN*mean/var 段（随参数落盘）。
    float* d_b[MAX_DEPTH];
    float* d_db[MAX_DEPTH];
    float* d_bn_mean[MAX_DEPTH];
    float* d_bn_var[MAX_DEPTH];
    float* d_params;    // [spec.total]         当前参数
    float* d_grad;      // [spec.total]         参数梯度（atomicAdd 累加，update 后清零）
    float* d_velocity;  // [spec.total]         SGD momentum 速度缓冲（训练态，不落盘）
    // B1：训练期累计统计（GPU 端 loss/acc，每 epoch 仅一次 D2H 取回打印）
    float* d_lossSum;   // [1]                  本 epoch loss 累加和
    int*   d_correct;   // [1]                  本 epoch 预测正确样本数
    int*   d_predCount; // [spec.classes]       本 epoch 各类预测计数
};

// ================= Kernel 声明 =================
// 阶段2：除 kgather 外，所有 kernel 末尾新增按值参数 const NetSpec N，
// 维度与参数偏移一律读 N.*，不再使用编译期宏。

// 按索引把全量数据 gather 成当前 batch
__global__ void kgather(const float* __restrict__ data,
                        const int*   __restrict__ allLabel,
                        const int*   __restrict__ index,
                        float*       __restrict__ outInput,
                        int*         __restrict__ outLabel,
                        int inSize);

// 前向：第 li 层 Conv（仅卷积，输出 d_c[li]；不含 ReLU/Pool），block-per-sample。
// 维度/偏移取自 N.layer[li] 与 N.offK[li]/N.offB[li]；输入为上一层池化输出（li=0 时为网络输入）。
__global__ void kconv(const float* __restrict__ input,
                      const float* __restrict__ params,
                      float* __restrict__ c,
                      int li,
                      const NetSpec N);

// 前向：第 li 层 ReLU + MaxPool（输入 preRelu：非 BN 传 d_c[li]，BN 传 BN 输出 d_b[li]）。
__global__ void krelu_pool(const float* __restrict__ preRelu,
                           float* __restrict__ r,
                           float* __restrict__ p,
                           int li,
                           const NetSpec N);

// ================= BN kernel（P2，仅 spec.bn==true 调用） =================
// BN 统计：每通道在 (batch×H×W) 上求「有偏」均值/方差。
//   x 布局 [batch][C][H][W] 连续；count = batch*H*W；HW = H*W；block-per-channel。
__global__ void kbn_stats(const float* __restrict__ x,
                          size_t count, int C, int HW,
                          float* __restrict__ mean_out,
                          float* __restrict__ var_out);

// BN 应用：y = gamma[c]*(x-mean[c])/sqrt(var[c]+eps)+beta[c]。
// 并行 over total = batch*C*HW；通道 c = (idx / HW) % C。
__global__ void kbn_apply(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          const float* __restrict__ mean,
                          const float* __restrict__ var,
                          float eps, size_t total, int C, int HW,
                          float* __restrict__ y);

// BN 反向：block-per-channel（grid=C）。先归约本批 dgamma/dbeta 并 atomicAdd 到
// grad 的 off_g/off_b 段，再计算 dx。x/dy/dx 布局 [batch][C][HW]。
__global__ void kbn_bwd(const float* __restrict__ x,
                        const float* __restrict__ dy,
                        const float* __restrict__ gamma,
                        const float* __restrict__ mean,
                        const float* __restrict__ var,
                        float eps, int C, int HW, int batch,
                        float* __restrict__ grad, int off_g, int off_b,
                        float* __restrict__ dx);

// BN running 统计更新：running = (1-m)*running + m*batch；
//   running var 用无偏 batch 方差（×N/(N-1)，N=count）。
__global__ void kbn_update_running(const float* __restrict__ batch_mean,
                                   const float* __restrict__ batch_var,
                                   float* __restrict__ running_mean,
                                   float* __restrict__ running_var,
                                   int C, size_t count, float momentum);

// 前向：FC + Softmax
__global__ void kf3(const float* __restrict__ p2,
                    const float* __restrict__ params,
                    float* __restrict__ logits,
                    float* __restrict__ prob,
                    const NetSpec N);

// B1：GPU 端 loss / 准确率累加。每 block 负责一个样本（grid = n），
// block 内先做 argmax 归约，仅 threadIdx.x==0 用 atomicAdd 累加，
// 从而消除训练循环中每 batch 的 D2H + host 串行计算。
__global__ void kloss_acc(const float* __restrict__ prob,
                          const int*   __restrict__ label,
                          int n,
                          float* __restrict__ d_lossSum,
                          int*   __restrict__ d_correct,
                          int*   __restrict__ d_predCount,
                          const NetSpec N);

// 反向：dlogits -> dW_fc / db_fc / dflat(=d_p2)
__global__ void kb_fc(const float* __restrict__ prob,
                      const int*   __restrict__ label,
                      const float* __restrict__ p2,
                      const float* __restrict__ params,
                      float* __restrict__ grad,
                      float* __restrict__ dflat,
                      const NetSpec N);

// 反向：第 li 层 Pool 反向 + ReLU 反向（融合）-> dc（out，即 d_dc[li] 或 BN 的 d_db[li]）。
// dpool 为该层池化输出梯度；preRelu 为 ReLU 输入（非 BN=d_c[li]，BN=d_b[li]）。
__global__ void kb_pool_relu(const float* __restrict__ dpool,
                             const float* __restrict__ r,
                             const float* __restrict__ preRelu,
                             float* __restrict__ dc,
                             int li,
                             const NetSpec N);

// 反向：第 li 层 Conv -> dK[li] / dB[li] / d(输入)（li==0 时 dpIn 传 nullptr，不需要 dinput）
// addMode==0：dpIn 覆盖写（旧行为，非残差网络逐位不变）；
// addMode==1：dpIn 累加写（残差网络，dpIn 可能已含其他层分流过来的梯度）。
__global__ void kb_conv(const float* __restrict__ input,
                        const float* __restrict__ dc,
                        const float* __restrict__ params,
                        float* __restrict__ grad,
                        float* __restrict__ dpIn,
                        int li,
                        const NetSpec N,
                        int addMode);

// 阶段3：残差加法 / 分流 kernel：dst[idx] += src[idx]（逐元素，block-per-sample）。
// nElem 为该张量每样本元素数（identity 残差要求 src/dst 形状一致，由 validateNetSpec 保证）。
__global__ void kadd(const float* __restrict__ src,
                     float* __restrict__ dst,
                     int nElem);

// 阶段6：1×1 投影残差前向：dst[idx] += Wproj*src + bproj（res==2）。
// 目标空间尺寸由 L.resPos 决定：resPos=0 用池化后尺寸；resPos=1 用卷积输出（ReLU 前）尺寸。
// 源层输出空间尺寸与目标一致（validateNetSpec 保证），故按 srcC×tH×tW 布局索引。
__global__ void kproj_add(const float* __restrict__ src,
                          const float* __restrict__ params,
                          float* __restrict__ dst,
                          int li,
                          const NetSpec N);

// 阶段6：1×1 投影残差反向：由 dy 求 dPW/dPB（atomicAdd 到 grad）与 d(src)（累加到 dsrcOut）。
// addMode==1 时 dsrcOut[idx] += dsrc（用于把梯度分流回 skip 源层）。
__global__ void kproj_bwd(const float* __restrict__ src,
                          const float* __restrict__ dy,
                          const float* __restrict__ params,
                          float* __restrict__ grad,
                          float* __restrict__ dsrcOut,
                          int li,
                          const NetSpec N,
                          int addMode);

// 参数更新 + 梯度清零（SGD with momentum，按参数区间选择 lr）
//   v = beta * v + g;  step = nesterov ? (g + beta * v) : v;  params -= lr * step
//   beta=0 时退化为朴素 SGD（step = g）。
// BN 的 running mean/var（统计量）经 isTrainableParam 判定后跳过：不更新、梯度清零。
// BN 的 γ/β 位于各自 conv 区间内，照常按 lr_conv1/lr_conv2 更新。
// wd：L2 weight decay 系数（默认 0.0）。仅作用于乘法权重（K/PW/FCW 段，见 isWeightDecayParam）；
//     非零时在平均梯度上叠加 wd*param（位保护：wd==0 不进叠加路径，逐位不变）。
__global__ void kb_update(float* __restrict__ params,
                          float* __restrict__ grad,
                          float* __restrict__ velocity,
                          float lr_conv1,
                          float lr_conv2,
                          float lr_fc,
                          int batch,
                          float beta,
                          int nesterov,
                          float wd,
                          const NetSpec N);
