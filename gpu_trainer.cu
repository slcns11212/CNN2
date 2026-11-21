// gpu_trainer.cu
// GPU 训练模块实现：显存管理、数据上传、主训练循环、对外接口 gpuTrain。
// 数据流：
//   训练前：H2D 上传全量数据/标签/参数，常驻显存
//   每 epoch：CPU shuffle 索引 -> H2D 索引
//   每 batch：gather 样本 -> 前向（kconv1/(kbn_stats+kbn_apply)/krelu_pool1 -> 同 conv2
//             -> kf3）-> kloss_acc(GPU 端累加 loss/acc) -> 反向（kb_fc -> kb_pool2_relu2 ->
//             [kbn_bwd] -> kb_conv2 -> kb_pool1_relu1 -> [kbn_bwd] -> kb_conv1）-> kb_update
//   BN 仅在 spec.bn==true 时插入对应 kernel；非 BN 路径逐位不变。
//   每 epoch 末：D2H 统计量与参数 -> 打印/回调（主程序负责落盘）
// 阶段2：所有维度/偏移来自传入的 NetSpec spec，不再使用编译期宏或 GPU_TOTAL_PARAMS。
#include "gpu_trainer.cuh"
#include "gpu_kernels.cuh"
#include "common.h"     // 共享常量与 formatDur（inline）

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>    // 跨平台计时（steady_clock，Windows/Linux 通用）
#include <utility>   // std::move（导出逐模型概率时转移大数组所有权）

// L1：每次 kernel launch 后立即检查异步启动错误（cudaGetLastError 会同时清除错误状态）。
// 仅用于返回 bool 的 launch 辅助函数/入口；失败时打印 kernel 名并 return false，
// 不再静默继续。注意：该宏依赖使用它的函数返回 bool。
#define CUDA_CHECK_KERNEL(what)                                               \
    do {                                                                      \
        cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "gpu: kernel %s launch failed: %s\n",        \
                         (what), cudaGetErrorString(_e));                     \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ================= 显存上下文 =================
struct GpuContext {
    GpuBuffers b = {};   // 全部指针初始化为 nullptr，release() 可安全调用

    ~GpuContext() { release(); }

    void release() {
        cudaFree(b.d_data);
        cudaFree(b.d_allLabel);
        cudaFree(b.d_index);
        cudaFree(b.d_input);
        cudaFree(b.d_label);
        for (int i = 0; i < MAX_DEPTH; ++i) {
            cudaFree(b.d_c[i]);
            cudaFree(b.d_r[i]);
            cudaFree(b.d_p[i]);
            cudaFree(b.d_dp[i]);
            cudaFree(b.d_dc[i]);
            // BN 缓冲（非 BN 时均为 nullptr，cudaFree(nullptr) 安全）
            cudaFree(b.d_b[i]);
            cudaFree(b.d_db[i]);
            cudaFree(b.d_bn_mean[i]);
            cudaFree(b.d_bn_var[i]);
        }
        cudaFree(b.d_logits);
        cudaFree(b.d_prob);
        cudaFree(b.d_dflat);
        cudaFree(b.d_params);
        cudaFree(b.d_grad);
        cudaFree(b.d_velocity);
        cudaFree(b.d_lossSum);
        cudaFree(b.d_correct);
        cudaFree(b.d_predCount);
        std::memset(&b, 0, sizeof(b));
    }

    // 按运行时规格 spec 逐层分配全部显存缓冲（N = 全量样本数，B = batch 大小）。
    bool alloc(size_t N, size_t B, const NetSpec& spec) {
        const size_t inSize = (size_t)spec.inH * (size_t)spec.inW;
        bool ok = true;
        ok = ok && (cudaMalloc((void**)&b.d_data,     N * inSize       * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_allLabel, N                * sizeof(int))   == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_index,    N                * sizeof(int))   == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_input,    B * inSize       * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_label,    B                * sizeof(int))   == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_logits,   B * spec.classes * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_prob,     B * spec.classes * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_dflat,    B * spec.flat    * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_params,   spec.total       * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_grad,     spec.total       * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_velocity, spec.total       * sizeof(float)) == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_lossSum,   sizeof(float))       == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_correct,   sizeof(int))         == cudaSuccess);
        ok = ok && (cudaMalloc((void**)&b.d_predCount, spec.classes * sizeof(int)) == cudaSuccess);
        // 逐层张量：act = outC*H*W（conv 输出/ReLU/BN），pool = 该层输出（pool 时 /2，否则同 act）
        for (int i = 0; i < spec.depth && i < MAX_DEPTH; ++i) {
            const LayerSpec& L = spec.layer[i];
            const size_t act  = (size_t)L.outC * (size_t)L.H * (size_t)L.W;
            const size_t poolH = L.pool ? (size_t)(L.H / 2) : (size_t)L.H;
            const size_t poolW = L.pool ? (size_t)(L.W / 2) : (size_t)L.W;
            const size_t pool = (size_t)L.outC * poolH * poolW;
            ok = ok && (cudaMalloc((void**)&b.d_c[i],  B * act  * sizeof(float)) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&b.d_r[i],  B * act  * sizeof(float)) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&b.d_p[i],  B * pool * sizeof(float)) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&b.d_dp[i], B * pool * sizeof(float)) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&b.d_dc[i], B * act  * sizeof(float)) == cudaSuccess);
            // BN 缓冲：仅 spec.bn==true 时分配（非 BN 路径完全不占用）
            if (spec.bn) {
                ok = ok && (cudaMalloc((void**)&b.d_b[i],        B * act * sizeof(float)) == cudaSuccess);
                ok = ok && (cudaMalloc((void**)&b.d_db[i],       B * act * sizeof(float)) == cudaSuccess);
                ok = ok && (cudaMalloc((void**)&b.d_bn_mean[i],  L.outC * sizeof(float))  == cudaSuccess);
                ok = ok && (cudaMalloc((void**)&b.d_bn_var[i],   L.outC * sizeof(float))  == cudaSuccess);
            }
        }
        if (!ok) {
            std::fprintf(stderr, "cudaMalloc failed (out of memory?)\n");
            release();
            return false;
        }
        return true;
    }
};

// ================= 设备可用性 =================
bool gpuAvailable() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        cudaGetLastError();   // 清除错误状态
        return false;
    }
    return n > 0;
}

// ================= 设备名称 =================
std::string gpuName() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        cudaGetLastError();                 // 清除可能的错误状态
        return "CPU";
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return "N/A";
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "N/A";
    return std::string(prop.name);
}

// ================= 前向 / 反向 launch 辅助（train / eval 分支） =================
// 训练前向：逐层 Conv→BN(用 batch 统计、并更新 running)→ReLU→Pool；再 FC+Softmax。
// 非 BN 路径仅 Conv→ReLU→Pool，不分配、不跑任何 BN kernel，逐位不变。
static bool launchForwardTrain(GpuBuffers& B, const NetSpec& spec, int n) {
    for (int i = 0; i < spec.depth; ++i) {
        const LayerSpec& L = spec.layer[i];
        const int hw = L.H * L.W;
        const float* in = (i == 0) ? B.d_input : B.d_p[i - 1];

        kconv<<<n, CNN_BLOCK>>>(in, B.d_params, B.d_c[i], i, spec);
        CUDA_CHECK_KERNEL("kconv");
        if (spec.bn) {
            // M3：count/total 以 size_t 计算，避免大 batch×大通道时 int 溢出
            const size_t cnt   = (size_t)n * (size_t)hw;
            const size_t total = (size_t)n * (size_t)L.outC * (size_t)hw;
            kbn_stats<<<L.outC, CNN_BLOCK>>>(B.d_c[i], cnt, L.outC, hw,
                                             B.d_bn_mean[i], B.d_bn_var[i]);
            CUDA_CHECK_KERNEL("kbn_stats");
            kbn_apply<<<(unsigned)((total + CNN_BLOCK - 1) / CNN_BLOCK), CNN_BLOCK>>>(
                B.d_c[i], B.d_params + spec.offBNg[i], B.d_params + spec.offBNb[i],
                B.d_bn_mean[i], B.d_bn_var[i], BN_EPS, total, L.outC, hw, B.d_b[i]);
            CUDA_CHECK_KERNEL("kbn_apply");
            kbn_update_running<<<(L.outC + CNN_BLOCK - 1) / CNN_BLOCK, CNN_BLOCK>>>(
                B.d_bn_mean[i], B.d_bn_var[i],
                B.d_params + spec.offBNmean[i], B.d_params + spec.offBNvar[i],
                L.outC, cnt, BN_MOMENTUM);
            CUDA_CHECK_KERNEL("kbn_update_running");
            // 阶段6：pre-activation 残差（resPos=1）在 BN 后、ReLU 前叠加到 d_b[i]。
            if (L.res != 0 && L.resPos == 1 && L.resFrom >= 0) {
                const int actSize = L.outC * L.H * L.W;
                if (L.res == 1) {
                    kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_b[i], actSize);
                    CUDA_CHECK_KERNEL("kadd(preact)");
                } else {
                    kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_b[i], i, spec);
                    CUDA_CHECK_KERNEL("kproj_add(preact)");
                }
            }
            krelu_pool<<<n, CNN_BLOCK>>>(B.d_b[i], B.d_r[i], B.d_p[i], i, spec);
        } else {
            // 阶段6：pre-activation 残差（resPos=1）在 ReLU 前叠加到 d_c[i]（非 BN 无 BN 输出）。
            if (L.res != 0 && L.resPos == 1 && L.resFrom >= 0) {
                const int actSize = L.outC * L.H * L.W;
                if (L.res == 1) {
                    kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_c[i], actSize);
                    CUDA_CHECK_KERNEL("kadd(preact)");
                } else {
                    kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_c[i], i, spec);
                    CUDA_CHECK_KERNEL("kproj_add(preact)");
                }
            }
            krelu_pool<<<n, CNN_BLOCK>>>(B.d_c[i], B.d_r[i], B.d_p[i], i, spec);
        }
        CUDA_CHECK_KERNEL("krelu_pool");
        // 阶段3/6：post 残差（resPos=0）在该层最终输出上叠加（identity=kadd；投影=kproj_add）。
        if (L.res != 0 && L.resPos == 0 && L.resFrom >= 0) {
            const int poolSize = L.outC * (L.pool ? (L.H / 2) : L.H) * (L.pool ? (L.W / 2) : L.W);
            if (L.res == 1) {
                kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_p[i], poolSize);
                CUDA_CHECK_KERNEL("kadd");
            } else {
                kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_p[i], i, spec);
                CUDA_CHECK_KERNEL("kproj_add");
            }
        }
    }

    kf3<<<n, CNN_BLOCK>>>(B.d_p[spec.depth - 1], B.d_params, B.d_logits, B.d_prob, spec);
    CUDA_CHECK_KERNEL("kf3");
    return true;
}

// 推理前向（eval 模式）：逐层 Conv→BN(用 running 统计量，**不更新** running)→ReLU→Pool。
// input 允许指向任意批起点（gpuPredict 直接以 d_data 偏移作为输入）。
static bool launchForwardEval(const float* input, GpuBuffers& B,
                              const NetSpec& spec, int n) {
    for (int i = 0; i < spec.depth; ++i) {
        const LayerSpec& L = spec.layer[i];
        const int hw = L.H * L.W;
        const float* in = (i == 0) ? input : B.d_p[i - 1];

        kconv<<<n, CNN_BLOCK>>>(in, B.d_params, B.d_c[i], i, spec);
        CUDA_CHECK_KERNEL("kconv");
        if (spec.bn) {
            // M3：total 以 size_t 计算，避免大 batch×大通道时 int 溢出
            const size_t total = (size_t)n * (size_t)L.outC * (size_t)hw;
            kbn_apply<<<(unsigned)((total + CNN_BLOCK - 1) / CNN_BLOCK), CNN_BLOCK>>>(
                B.d_c[i], B.d_params + spec.offBNg[i], B.d_params + spec.offBNb[i],
                B.d_params + spec.offBNmean[i], B.d_params + spec.offBNvar[i],
                BN_EPS, total, L.outC, hw, B.d_b[i]);
            CUDA_CHECK_KERNEL("kbn_apply");
            if (L.res != 0 && L.resPos == 1 && L.resFrom >= 0) {
                const int actSize = L.outC * L.H * L.W;
                if (L.res == 1) {
                    kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_b[i], actSize);
                    CUDA_CHECK_KERNEL("kadd(preact)");
                } else {
                    kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_b[i], i, spec);
                    CUDA_CHECK_KERNEL("kproj_add(preact)");
                }
            }
            krelu_pool<<<n, CNN_BLOCK>>>(B.d_b[i], B.d_r[i], B.d_p[i], i, spec);
        } else {
            if (L.res != 0 && L.resPos == 1 && L.resFrom >= 0) {
                const int actSize = L.outC * L.H * L.W;
                if (L.res == 1) {
                    kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_c[i], actSize);
                    CUDA_CHECK_KERNEL("kadd(preact)");
                } else {
                    kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_c[i], i, spec);
                    CUDA_CHECK_KERNEL("kproj_add(preact)");
                }
            }
            krelu_pool<<<n, CNN_BLOCK>>>(B.d_c[i], B.d_r[i], B.d_p[i], i, spec);
        }
        CUDA_CHECK_KERNEL("krelu_pool");
        // 阶段3/6：eval 前向支持 identity/投影残差（test --gpu 加载残差模型）。
        if (L.res != 0 && L.resPos == 0 && L.resFrom >= 0) {
            const int poolSize = L.outC * (L.pool ? (L.H / 2) : L.H) * (L.pool ? (L.W / 2) : L.W);
            if (L.res == 1) {
                kadd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_p[i], poolSize);
                CUDA_CHECK_KERNEL("kadd");
            } else {
                kproj_add<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_params, B.d_p[i], i, spec);
                CUDA_CHECK_KERNEL("kproj_add");
            }
        }
    }

    kf3<<<n, CNN_BLOCK>>>(B.d_p[spec.depth - 1], B.d_params, B.d_logits, B.d_prob, spec);
    CUDA_CHECK_KERNEL("kf3");
    return true;
}

// 反向：逐层从末层到首层组织调用链。BN 时以 d_b[i] 作为 ReLU 输入做 mask，
// 并在 pool 反向与 conv 反向之间插入 kbn_bwd 求 dx 及累加 dγ/dβ。
// 阶段3：残差网络（hasRes）下，d_dp[i] 作为「第 i 层池化输出梯度」全集缓冲，
//   由后续层主干反传与残差分流共同累加；非残差网络保持旧路径与逐位数值不变。
static bool launchBackward(GpuBuffers& B, const NetSpec& spec, int n) {
    bool hasRes = false;
    for (int i = 0; i < spec.depth; ++i)
        if (spec.layer[i].res != 0) { hasRes = true; break; }

    // fc 反向：dflat = 末层池化输出梯度
    kb_fc<<<n, CNN_BLOCK>>>(B.d_prob, B.d_label, B.d_p[spec.depth - 1], B.d_params,
                            B.d_grad, B.d_dflat, spec);
    CUDA_CHECK_KERNEL("kb_fc");

    // 残差网络：把所有 d_dp[i] 清零，再把末层梯度 d_dflat 放入 d_dp[last]。
    if (hasRes) {
        for (int i = 0; i < spec.depth; ++i) {
            const LayerSpec& L = spec.layer[i];
            const int ph = L.pool ? (L.H / 2) : L.H;
            const int pw = L.pool ? (L.W / 2) : L.W;
            cudaError_t me = cudaMemsetAsync(B.d_dp[i], 0,
                (size_t)n * L.outC * ph * pw * sizeof(float));
            if (me != cudaSuccess) {
                std::fprintf(stderr, "gpu: cudaMemsetAsync(d_dp) failed: %s\n",
                             cudaGetErrorString(me));
                return false;
            }
        }
        const LayerSpec& lastL = spec.layer[spec.depth - 1];
        const int lph = lastL.pool ? (lastL.H / 2) : lastL.H;
        const int lpw = lastL.pool ? (lastL.W / 2) : lastL.W;
        cudaError_t ce = cudaMemcpyAsync(B.d_dp[spec.depth - 1], B.d_dflat,
            (size_t)n * lastL.outC * lph * lpw * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "gpu: cudaMemcpyAsync(dflat->d_dp) failed: %s\n",
                         cudaGetErrorString(ce));
            return false;
        }
    }

    for (int i = spec.depth - 1; i >= 0; --i) {
        const LayerSpec& L = spec.layer[i];
        const int actSize = L.outC * L.H * L.W;
        const int ph = L.pool ? (L.H / 2) : L.H;
        const int pw = L.pool ? (L.W / 2) : L.W;
        const int poolSize = L.outC * ph * pw;

        // 残差分流（resPos=0：梯度从最终输出 d_dp[i] 分流，须在消费 d_dp[i] 之前）。
        // identity 用 kadd；投影用 kproj_bwd（同时累加 dPW/dPB 并把 d(src) 加回源层）。
        if (hasRes && L.res != 0 && L.resPos == 0 && L.resFrom >= 0) {
            if (L.res == 1) {
                kadd<<<n, CNN_BLOCK>>>(B.d_dp[i], B.d_dp[L.resFrom], poolSize);
                CUDA_CHECK_KERNEL("kadd");
            } else {
                kproj_bwd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], B.d_dp[i], B.d_params,
                                            B.d_grad, B.d_dp[L.resFrom], i, spec, 1);
                CUDA_CHECK_KERNEL("kproj_bwd");
            }
        }

        // 本层池化输出梯度：残差网络统一取 d_dp[i]；非残差沿用旧路径（末层 dflat）。
        const float* poolGrad = hasRes
            ? B.d_dp[i]
            : ((i == spec.depth - 1) ? B.d_dflat : B.d_dp[i]);
        const float* preRelu  = spec.bn ? B.d_b[i] : B.d_c[i];
        // pool/relu 反向输出：BN 时写 d_db[i]（再经 kbn_bwd 转 d_c[i]），否则直接写 d_c[i]
        float* poolOut = spec.bn ? B.d_db[i] : B.d_dc[i];

        // L1：cudaMemsetAsync 返回值必须检查，失败不再静默继续
        cudaError_t me = cudaMemsetAsync(poolOut, 0, (size_t)n * actSize * sizeof(float));
        if (me != cudaSuccess) {
            std::fprintf(stderr, "gpu: cudaMemsetAsync(poolGrad) failed: %s\n",
                         cudaGetErrorString(me));
            return false;
        }
        kb_pool_relu<<<n, CNN_BLOCK>>>(poolGrad, B.d_r[i], preRelu, poolOut, i, spec);
        CUDA_CHECK_KERNEL("kb_pool_relu");

        // 残差分流（resPos=1：梯度从 ReLU 前的 poolOut=d_db[i] 或 d_dc[i] 分流）。
        if (hasRes && L.res != 0 && L.resPos == 1 && L.resFrom >= 0) {
            if (L.res == 1) {
                kadd<<<n, CNN_BLOCK>>>(poolOut, B.d_dp[L.resFrom], actSize);
                CUDA_CHECK_KERNEL("kadd(preact)");
            } else {
                kproj_bwd<<<n, CNN_BLOCK>>>(B.d_p[L.resFrom], poolOut, B.d_params,
                                            B.d_grad, B.d_dp[L.resFrom], i, spec, 1);
                CUDA_CHECK_KERNEL("kproj_bwd(preact)");
            }
        }

        if (spec.bn) {
            kbn_bwd<<<L.outC, CNN_BLOCK>>>(B.d_c[i], B.d_db[i],
                B.d_params + spec.offBNg[i], B.d_bn_mean[i], B.d_bn_var[i], BN_EPS,
                L.outC, L.H * L.W, n, B.d_grad, spec.offBNg[i], spec.offBNb[i], B.d_dc[i]);
            CUDA_CHECK_KERNEL("kbn_bwd");
        }

        const float* convIn = (i == 0) ? B.d_input : B.d_p[i - 1];
        float* dpIn = (i == 0) ? nullptr : B.d_dp[i - 1];
        // 残差网络：dpIn 需累加（可能已有分流项）；非残差：覆盖写（逐位不变）。
        kb_conv<<<n, CNN_BLOCK>>>(convIn, B.d_dc[i], B.d_params, B.d_grad, dpIn, i, spec,
                                  hasRes ? 1 : 0);
        CUDA_CHECK_KERNEL("kb_conv");
    }
    return true;
}

// ================= 主训练入口 =================
bool gpuTrain(const std::vector<float>& data,
              const std::vector<int>& labels,
              const GpuTrainConfig& cfg,
              std::vector<float>& params,
              GpuEpochCallback onEpochEnd,
              void* userData,
              const NetSpec& spec)
{
    const size_t N = labels.size();
    const size_t inSize = (size_t)spec.inH * (size_t)spec.inW;
    if (N == 0 || data.size() != N * inSize) {
        std::fprintf(stderr, "gpuTrain: invalid data/label size\n");
        return false;
    }
    if (params.size() != (size_t)spec.total) {
        std::fprintf(stderr, "gpuTrain: params size must be %d\n", spec.total);
        return false;
    }

    int batch = cfg.batch;
    if (batch < 1) batch = 1;
    if ((size_t)batch > N) batch = (int)N;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "gpuTrain: cudaSetDevice failed\n");
        return false;
    }

    GpuContext ctx;
    if (!ctx.alloc(N, (size_t)batch, spec))
        return false;
    GpuBuffers& B = ctx.b;

    // ---- 一次性上传全量数据 / 标签 / 参数，梯度与 momentum 速度清零 ----
    if (cudaMemcpy(B.d_data, data.data(), N * inSize * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(B.d_allLabel, labels.data(), N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(B.d_params, params.data(), spec.total * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(B.d_grad, 0, spec.total * sizeof(float)) != cudaSuccess ||
        cudaMemset(B.d_velocity, 0, spec.total * sizeof(float)) != cudaSuccess)
    {
        std::fprintf(stderr, "gpuTrain: initial upload failed\n");
        return false;
    }

    // ---- 每个区间基准学习率：被禁用的组置 0（权重不变） ----
    // one-cycle 在此基础上乘以 lrScaleFactor；默认 oneCycle=false 时因子恒为 1.0f，
    // base*1.0f 在 IEEE754 下精确等于 base，保证与固定 LR 逐位一致。
    const float base_conv1 = cfg.train_conv1 ? cfg.lr_conv1 : 0.0f;
    const float base_conv2 = cfg.train_conv2 ? cfg.lr_conv2 : 0.0f;
    const float base_fc    = cfg.train_fc    ? cfg.lr_fc    : 0.0f;

    // ---- one-cycle 调度步数（per-batch 粒度） ----
    const long long stepsPerEpoch =
        (long long)((N + (size_t)batch - 1) / (size_t)batch);
    const long long totalSteps = (long long)cfg.epochs * stepsPerEpoch;
    const int warmEpochs = (cfg.lrSched.warmupEpochs > 0)
        ? cfg.lrSched.warmupEpochs
        : std::max(1, cfg.epochs / 5);
    const long long warmupSteps = (long long)warmEpochs * stepsPerEpoch;

    const int updBlocks = (spec.total + CNN_BLOCK - 1) / CNN_BLOCK;

    std::vector<int> order(N);
    for (size_t i = 0; i < N; ++i) order[i] = (int)i;

    // 阶段4.1：每 epoch 用 (seed, epoch) 派生的确定性引擎洗牌，替换 random_device。
    // 同 seed 同数据 → 每 epoch 洗牌顺序固定，训练逐位可复现。
    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        // 跨平台计时：epoch 起点（含 shuffle 与索引上传）
        auto tEp0 = std::chrono::steady_clock::now();
        std::mt19937 rng(mixSeed(cfg.seed, 0x5EEDu ^ (unsigned)epoch));
        std::shuffle(order.begin(), order.end(), rng);

        if (cudaMemcpy(B.d_index, order.data(), N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "gpuTrain: index upload failed\n");
            return false;
        }

        long long total = 0;
        // B1：本 epoch 的 loss/correct/predCount 在 GPU 上累加，epoch 开始前清零一次
        if (cudaMemset(B.d_lossSum, 0, sizeof(float)) != cudaSuccess ||
            cudaMemset(B.d_correct, 0, sizeof(int)) != cudaSuccess ||
            cudaMemset(B.d_predCount, 0, spec.classes * sizeof(int)) != cudaSuccess) {
            std::fprintf(stderr, "gpuTrain: loss accumulator reset failed\n");
            return false;
        }

        for (size_t start = 0; start < N; start += batch) {
            const int n = (int)std::min((size_t)batch, N - start);
            const long long batchIndex = (long long)(start / (size_t)batch);

            // ---- gather 当前 batch ----
            kgather<<<n, CNN_BLOCK>>>(B.d_data, B.d_allLabel, B.d_index + start,
                                      B.d_input, B.d_label, (int)inSize);
            CUDA_CHECK_KERNEL("kgather");

            // ---- 前向（train 模式：若 BN，则用 batch 统计并更新 running） ----
            if (!launchForwardTrain(B, spec, n)) return false;

            // ---- B1：GPU 端累加 loss / acc / 每类预测计数（无每 batch D2H 同步） ----
            // d_label 已由 kgather 写入本 batch 标签，等价于 labels[order[start+s]]
            kloss_acc<<<n, CNN_BLOCK>>>(B.d_prob, B.d_label, n,
                                        B.d_lossSum, B.d_correct, B.d_predCount, spec);
            CUDA_CHECK_KERNEL("kloss_acc");
            total += n;

            // ---- 反向（BN 时在 pool 反向与 conv 反向之间插入 kbn_bwd；含必要的 memset） ----
            if (!launchBackward(B, spec, n)) return false;

            // ---- 更新 + 梯度清零（SGD with momentum） ----
            // grad 为整个 batch 累加结果，传入当前 batch 实际样本数 n 做平均。
            // one-cycle：按当前全局 step 计算本 batch 的 lr 倍率（默认恒 1.0f）。
            const long long step = (long long)epoch * stepsPerEpoch + batchIndex;
            const float lrF = lrScaleFactor(cfg.lrSched, step, totalSteps, warmupSteps);
            const float lr_conv1 = base_conv1 * lrF;
            const float lr_conv2 = base_conv2 * lrF;
            const float lr_fc    = base_fc    * lrF;
            kb_update<<<updBlocks, CNN_BLOCK>>>(B.d_params, B.d_grad, B.d_velocity,
                                                lr_conv1, lr_conv2, lr_fc, n,
                                                cfg.momentum, cfg.nesterov ? 1 : 0,
                                                cfg.weightDecay, spec);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gpuTrain: kernel launch failed: %s\n", cudaGetErrorString(err));
                return false;
            }
        }

        if (total == 0) {
            std::fprintf(stderr, "gpuTrain: no valid training samples\n");
            return false;
        }

        // ---- 本 epoch 唯一一次统计 D2H：1 个 float + 1 个 int + spec.classes 个 int ----
        float h_lossSum = 0.0f;
        int   h_correct = 0;
        std::vector<int> h_predCount(spec.classes, 0);
        if (cudaMemcpy(&h_lossSum, B.d_lossSum, sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&h_correct, B.d_correct, sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_predCount.data(), B.d_predCount, spec.classes * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "gpuTrain: loss download failed\n");
            return false;
        }

        // 阶段4.3：把本 epoch 统计量先落为局部变量，供 stdout 打印与回调（--log）共用，
        // 保证日志列与打印数值同源。默认路径 stdout 逐字节不变。
        const double lossAvg = (double)h_lossSum / (double)total;
        const float  acc     = (float)h_correct / (float)total;
        // 本 epoch 末（最后一个 step）的 lr：默认 one-cycle 关闭时 fEp=1.0f，lr 即 base。
        const long long epReportStep = (long long)epoch * stepsPerEpoch
            + (stepsPerEpoch > 0 ? stepsPerEpoch - 1 : 0);
        const float fEp        = lrScaleFactor(cfg.lrSched, epReportStep, totalSteps, warmupSteps);
        const float epLrConv1  = base_conv1 * fEp;
        const float epLrConv2  = base_conv2 * fEp;
        const float epLrFc     = base_fc    * fEp;

        std::cout << "epoch " << epoch
                  << " loss " << lossAvg
                  << " acc " << acc;
        // 仅 one-cycle 开启时追加 lr，保证默认 stdout 与改动前逐字节一致
        if (cfg.lrSched.oneCycle) {
            // 本 epoch 末（最后一个 step）的 lr，便于核对 one-cycle 曲线
            std::cout << " lr(c1/c2/fc)=" << epLrConv1
                      << "/" << epLrConv2
                      << "/" << epLrFc;
        }
        std::cout << "\n";
        std::cout << "pred dist: ";
        for (int i = 0; i < spec.classes; ++i)
            std::cout << i << ":" << h_predCount[i] << " ";
        std::cout << "\n";

        // ---- epoch 末：D2H 参数，交由主程序落盘 ----
        // 注意：该 D2H 拷贝是同步操作，会等待本 epoch 所有异步 kernel 执行完毕，
        // 因此其返回后取时间戳可覆盖真实 GPU 执行时间（不会漏计异步 kernel）。
        if (cudaMemcpy(params.data(), B.d_params, spec.total * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "gpuTrain: param download failed\n");
            return false;
        }
        // --time：每个 epoch 耗时（此前的 D2H 参数拷贝是同步操作，
        // 返回后本 epoch 所有异步 kernel 均已执行完毕，计时准确）
        // 阶段4.3：耗时无论是否 --time 都算好传给回调，供 --log 的 epoch_sec 列使用。
        const double epSec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tEp0).count();
        if (cfg.timeIt) {
            std::cout << "epoch " << epoch << " 耗时: " << formatDur(epSec) << "\n";
        }
        if (onEpochEnd)
            onEpochEnd(params.data(), epoch, (float)lossAvg, acc,
                       epLrConv1, epLrConv2, epLrFc, epSec, userData);
    }

    // ---- 最终再同步一次参数（cfg.epochs<=0 时也能拿到初值） ----
    if (cudaMemcpy(params.data(), B.d_params, spec.total * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "gpuTrain: final param download failed\n");
        return false;
    }

    // ---- 最终同步：检查反向/更新 kernel 的异步运行期错误（越界/非法访问） ----
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "gpuTrain: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncErr));
        return false;
    }
    return true;
}

// ================= 批量前向推理（test --gpu 用，支持多模型概率平均集成） =================
// 只跑前向（kconv1/krelu_pool1、kconv2/krelu_pool2、kf3；BN 时用 eval 模式）：数据按顺序连续存放，直接指针偏移替代 gather。
// 对 paramsList 中每个模型：上传其参数 -> 跑全测试集前向 -> D2H 概率并累加。
// 显存只用一个 d_params 缓冲，逐模型覆盖上传，避免多份权重常驻。
// 最后对每样本的累加概率取 argmax（除以 M 不影响 argmax，规则与 CPU 完全一致）。
bool gpuPredict(const std::vector<float>& data,
                int N,
                const std::vector<std::vector<float>>& paramsList,
                int batch,
                std::vector<int>& outPred,
                std::vector<float>* outAllProb,
                const NetSpec& spec)
{
    const size_t inSize = (size_t)spec.inH * (size_t)spec.inW;
    if (N <= 0 || data.size() != (size_t)N * inSize) {
        std::fprintf(stderr, "gpuPredict: invalid data size\n");
        return false;
    }
    if (paramsList.empty()) {
        std::fprintf(stderr, "gpuPredict: empty params list\n");
        return false;
    }
    for (size_t m = 0; m < paramsList.size(); ++m) {
        if (paramsList[m].size() != (size_t)spec.total) {
            std::fprintf(stderr, "gpuPredict: params[%zu] size must be %d\n",
                         m, spec.total);
            return false;
        }
    }
    if (batch < 1) batch = 1;
    if ((size_t)batch > (size_t)N) batch = N;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "gpuPredict: cudaSetDevice failed\n");
        return false;
    }

    // 复用训练用的完整批缓冲（只用到其中的前向部分），逻辑简单且不会越界
    GpuContext ctx;
    if (!ctx.alloc((size_t)N, (size_t)batch, spec))
        return false;
    GpuBuffers& B = ctx.b;

    if (cudaMemcpy(B.d_data, data.data(), (size_t)N * inSize * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::fprintf(stderr, "gpuPredict: initial upload failed\n");
        return false;
    }

    // probSum[i*spec.classes + c] = Σ_m prob_m(c | sample i)
    std::vector<float> probSum((size_t)N * spec.classes, 0.0f);
    std::vector<float> hostProb((size_t)batch * spec.classes);

    // allProb[((m*N + i)*spec.classes) + c] = 第 m 个模型对第 i 个样本第 c 类的概率。
    // outAllProb 为 nullptr 时 allProbPtr 恒为 nullptr，写入分支被跳过（零开销）。
    const size_t M = paramsList.size();
    std::vector<float> allProb;
    float* allProbPtr = nullptr;
    if (outAllProb) {
        allProb.assign((size_t)N * M * spec.classes, 0.0f);
        allProbPtr = allProb.data();
    }

    for (size_t m = 0; m < paramsList.size(); ++m) {
        // 逐个模型上传到同一 d_params 缓冲（覆盖上一个模型）
        if (cudaMemcpy(B.d_params, paramsList[m].data(), spec.total * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "gpuPredict: params upload failed (model %zu)\n", m);
            return false;
        }

        for (int start = 0; start < N; start += batch) {
            const int n = std::min(batch, N - start);

            // 数据已按 [sample][inH*inW] 顺序连续存放，偏移指针即可作为 batch 输入
            const float* in = B.d_data + (size_t)start * inSize;
            // eval 模式：BN 用 running 统计量、不更新 running
            if (!launchForwardEval(in, B, spec, n)) return false;

            // 冗余复核（launchForwardEval 内部已逐 launch 检查 cudaGetLastError）
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gpuPredict: kernel launch failed: %s\n",
                             cudaGetErrorString(err));
                return false;
            }

            if (cudaMemcpy(hostProb.data(), B.d_prob,
                           (size_t)n * spec.classes * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "gpuPredict: prob download failed\n");
                return false;
            }
            for (int s = 0; s < n; ++s) {
                const float* pr = hostProb.data() + (size_t)s * spec.classes;
                const int i = start + s;  // 批内偏移 s -> 全局样本下标 i
                float* acc = probSum.data() + (size_t)i * spec.classes;
                if (allProbPtr) {
                    // 逐模型概率落位：((size_t)m * N + i) * spec.classes + c
                    float* dst = allProbPtr + ((size_t)m * N + i) * spec.classes;
                    for (int c = 0; c < spec.classes; ++c) {
                        acc[c] += pr[c];
                        dst[c] = pr[c];
                    }
                } else {
                    for (int c = 0; c < spec.classes; ++c) acc[c] += pr[c];
                }
            }
        }
    }

    // 平均概率 argmax：除以模型数不改变大小关系，直接比较累加和即可。
    outPred.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        const float* acc = probSum.data() + (size_t)i * spec.classes;
        outPred[i] = argmaxIndex(acc, spec.classes);
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "gpuPredict: cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(syncErr));
        return false;
    }

    // 可选导出逐模型概率（N * M * spec.classes），供融合/TTA 使用
    if (outAllProb) *outAllProb = std::move(allProb);
    return true;
}
