// gpu_kernels.cu
// 10 个 CUDA kernel 的实现（前向 3 + 反向 5 + 更新 1 + GPU 端 loss 1），
// 另含 gather 辅助 kernel。
// 数值语义严格复刻 CPU 版本：MaxPool 使用严格大于 '>' 且按固定顺序串行比较；
// Softmax 先减 max 再 exp；dlogits = prob - onehot；loss = -log(prob[label]+1e-8)。
// 阶段2：所有维度与参数偏移均取自传入的 NetSpec N，不再使用编译期宏。
#include "gpu_kernels.cuh"

// ================= gather：按索引取样本到批缓冲 =================
__global__ void kgather(const float* __restrict__ data,
                        const int*   __restrict__ allLabel,
                        const int*   __restrict__ index,
                        float*       __restrict__ outInput,
                        int*         __restrict__ outLabel,
                        int inSize)
{
    const int b = blockIdx.x;
    const int src = index[b];
    const float* s = data + (size_t)src * inSize;
    float* d = outInput + (size_t)b * inSize;
    for (int i = threadIdx.x; i < inSize; i += blockDim.x)
        d[i] = s[i];
    if (threadIdx.x == 0)
        outLabel[b] = allLabel[src];
}

// ================= kconv：第 li 层 Conv（仅卷积，输出 d_c[li]） =================
// 维度/偏移全部取自 N.layer[li] 与 N.offK[li]/N.offB[li]；stride/pad 取自层规格。
// li==0 输入为网络输入（inH×inW×1）；li>0 输入为上一层输出（pool 时 /2，pool=0 时原尺寸）。
// 非 BN 路径：kconv(li=0)+krelu_pool(li=0)+... 的运算顺序与浮点结果与改造前逐位一致。
__global__ void kconv(const float* __restrict__ input,
                      const float* __restrict__ params,
                      float* __restrict__ c,
                      int li,
                      const NetSpec N)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    const int iH = netLayerInH(N, li);
    const int iW = netLayerInW(N, li);
    const int inSize  = L.inC * iH * iW;
    const int outSize = L.outC * L.H * L.W;
    const float* in = input + (size_t)b * inSize;
    const float* K = params + N.offK[li];
    const float* B = params + N.offB[li];
    float* cb = c + (size_t)b * outSize;

    for (int idx = threadIdx.x; idx < outSize; idx += blockDim.x) {
        const int oc = idx / (L.H * L.W);
        const int rem = idx - oc * (L.H * L.W);
        const int oh = rem / L.W;
        const int ow = rem - oh * L.W;

        float sum = B[oc];
        for (int ic = 0; ic < L.inC; ++ic) {
            const float* inch = in + ic * (iH * iW);
            const float* k = K + oc * (L.inC * L.kH * L.kW) + ic * (L.kH * L.kW);
#pragma unroll
            for (int kh = 0; kh < L.kH; ++kh) {
                const int ih = oh * L.stride + kh - L.pad;
                if (ih < 0 || ih >= iH) continue;
#pragma unroll
                for (int kw = 0; kw < L.kW; ++kw) {
                    const int iw = ow * L.stride + kw - L.pad;
                    if (iw < 0 || iw >= iW) continue;
                    sum += inch[ih * iW + iw] * k[kh * L.kW + kw];
                }
            }
        }
        cb[idx] = sum;
    }
}

// ================= krelu_pool：第 li 层 ReLU + [可选] MaxPool =================
// preRelu 为 ReLU 的输入（非 BN：d_c[li]；BN：d_b[li]）。
// L.pool==0 时不上池化，p 与 r 同形（尺寸 outC*H*W）；否则池化到 outC*(H/2)*(W/2)。
__global__ void krelu_pool(const float* __restrict__ preRelu,
                           float* __restrict__ r,
                           float* __restrict__ p,
                           int li,
                           const NetSpec N)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    const int actSize = L.outC * L.H * L.W;
    const int pH = L.pool ? (L.H / 2) : L.H;
    const int pW = L.pool ? (L.W / 2) : L.W;
    const int poolSize = L.outC * pH * pW;
    const float* xb = preRelu + (size_t)b * actSize;
    float* rb = r + (size_t)b * actSize;
    float* pb = p + (size_t)b * poolSize;

    // ReLU（与 CPU relu 一致：x<0 -> 0，否则 x）
    for (int idx = threadIdx.x; idx < actSize; idx += blockDim.x) {
        const float x = xb[idx];
        rb[idx] = (x < 0.0f) ? 0.0f : x;
    }
    __syncthreads();

    if (!L.pool) {
        // pool=0：p 直接等于 ReLU 输出（逐元素拷贝，保持既有缓冲语义）
        for (int idx = threadIdx.x; idx < poolSize; idx += blockDim.x)
            pb[idx] = rb[idx];
        return;
    }

    // MaxPool（outC*pH*pW），严格大于、固定顺序
    for (int idx = threadIdx.x; idx < poolSize; idx += blockDim.x) {
        const int c = idx / (pH * pW);
        const int rem = idx - c * (pH * pW);
        const int oh = rem / pW;
        const int ow = rem - oh * pW;
        const int ih = oh * 2, iw = ow * 2;

        const float* rr = rb + c * (L.H * L.W);
        float m = rr[ih * L.W + iw];
        if (rr[ih * L.W + iw + 1] > m) m = rr[ih * L.W + iw + 1];
        if (rr[(ih + 1) * L.W + iw] > m) m = rr[(ih + 1) * L.W + iw];
        if (rr[(ih + 1) * L.W + iw + 1] > m) m = rr[(ih + 1) * L.W + iw + 1];
        pb[idx] = m;
    }
}

// ================= kbn_stats：每通道 batch 有偏均值/方差 =================
// x 布局 [batch][C][H][W] 连续；count = batch*H*W；HW = H*W。
// block-per-channel；两遍归约（先均值、再平方偏差），数值稳定。
__global__ void kbn_stats(const float* __restrict__ x,
                          size_t count, int C, int HW,
                          float* __restrict__ mean_out,
                          float* __restrict__ var_out)
{
    const int c = blockIdx.x;
    if (c >= C) return;
    const int t = threadIdx.x;
    __shared__ float s[CNN_BLOCK];

    // 第一遍：求 sum（M3：k/count 以 size_t 迭代，避免大 batch×大 HW 时 int 溢出）
    float sum = 0.0f;
    for (size_t k = (size_t)t; k < count; k += blockDim.x) {
        const size_t b = k / (size_t)HW, i = k - b * (size_t)HW;
        sum += x[(b * (size_t)C + c) * (size_t)HW + i];
    }
    s[t] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    __shared__ float sMean;
    if (t == 0) {
        sMean = s[0] / (float)count;
        mean_out[c] = sMean;
    }
    __syncthreads();

    // 第二遍：求平方偏差和 -> 有偏方差
    const float m = sMean;
    float sq = 0.0f;
    for (size_t k = (size_t)t; k < count; k += blockDim.x) {
        const size_t b = k / (size_t)HW, i = k - b * (size_t)HW;
        const float d = x[(b * (size_t)C + c) * (size_t)HW + i] - m;
        sq += d * d;
    }
    s[t] = sq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) var_out[c] = s[0] / (float)count;
}

// ================= kbn_apply：逐元素归一化 =================
// y = gamma[c]*(x-mean[c])/sqrt(var[c]+eps)+beta[c]；并行 over total=batch*C*HW。
__global__ void kbn_apply(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          const float* __restrict__ mean,
                          const float* __restrict__ var,
                          float eps, size_t total, int C, int HW,
                          float* __restrict__ y)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int c = (int)((idx / (size_t)HW) % (size_t)C);
        const float inv = 1.0f / sqrtf(var[c] + eps);
        y[idx] = gamma[c] * (x[idx] - mean[c]) * inv + beta[c];
    }
}

// ================= kbn_bwd：BN 反向（dx + dgamma/dbeta） =================
// block-per-channel（grid=C）。归约本批 dbeta=Σdy、dgamma=Σdy*xhat，
// atomicAdd 到 grad 的 off_b/off_g 通道段；再用公式求 dx：
//   dx = gamma*inv_std/N * (N*dy - Σdy - xhat*Σ(dy*xhat))
__global__ void kbn_bwd(const float* __restrict__ x,
                        const float* __restrict__ dy,
                        const float* __restrict__ gamma,
                        const float* __restrict__ mean,
                        const float* __restrict__ var,
                        float eps, int C, int HW, int batch,
                        float* __restrict__ grad, int off_g, int off_b,
                        float* __restrict__ dx)
{
    const int c = blockIdx.x;
    if (c >= C) return;
    const int t = threadIdx.x;
    // M3：count 以 size_t 计算，避免大 batch×大 HW 的 int 溢出
    const size_t count = (size_t)batch * (size_t)HW;

    __shared__ float sdy[CNN_BLOCK];
    __shared__ float sdg[CNN_BLOCK];
    const float inv_std = 1.0f / sqrtf(var[c] + eps);
    const float m = mean[c];

    float a = 0.0f, g = 0.0f;
    for (size_t k = (size_t)t; k < count; k += blockDim.x) {
        const size_t b = k / (size_t)HW, i = k - b * (size_t)HW;
        const size_t off = (b * (size_t)C + c) * (size_t)HW + i;
        const float d = dy[off];
        a += d;
        g += d * (x[off] - m) * inv_std;
    }
    sdy[t] = a; sdg[t] = g;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            sdy[t] += sdy[t + stride];
            sdg[t] += sdg[t + stride];
        }
        __syncthreads();
    }

    __shared__ float sumDy, sumDg;
    if (t == 0) {
        sumDy = sdy[0];
        sumDg = sdg[0];
        atomicAdd(&grad[off_g + c], sumDg);
        atomicAdd(&grad[off_b + c], sumDy);
    }
    __syncthreads();

    const float invN = 1.0f / (float)count;
    const float gy = gamma[c];
    const float z = sumDy, w = sumDg;
    for (size_t k = (size_t)t; k < count; k += blockDim.x) {
        const size_t b = k / (size_t)HW, i = k - b * (size_t)HW;
        const size_t off = (b * (size_t)C + c) * (size_t)HW + i;
        const float xhat = (x[off] - m) * inv_std;
        const float d = dy[off];
        dx[off] = gy * inv_std * invN * ((float)count * d - z - xhat * w);
    }
}

// ================= kbn_update_running：更新 running 统计量 =================
// running = (1-m)*running + m*batch；running var 用无偏（×N/(N-1)）。
__global__ void kbn_update_running(const float* __restrict__ batch_mean,
                                   const float* __restrict__ batch_var,
                                   float* __restrict__ running_mean,
                                   float* __restrict__ running_var,
                                   int C, size_t count, float momentum)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const size_t n1 = (count > 1) ? (count - 1) : 1;
    const float unbiased = batch_var[c] * ((float)count / (float)n1);
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * batch_mean[c];
    running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * unbiased;
}

// ================= kf3：FC + Softmax =================
__global__ void kf3(const float* __restrict__ p2,
                    const float* __restrict__ params,
                    float* __restrict__ logits,
                    float* __restrict__ prob,
                    const NetSpec N)
{
    const int b = blockIdx.x;
    const float* p2b = p2 + (size_t)b * N.flat;
    const float* W = params + N.offFCW;
    const float* bfc = params + N.offFCB;
    float* lo = logits + (size_t)b * N.classes;
    float* pr = prob + (size_t)b * N.classes;

    // M1 防御：每个 logits 槽位只由 threadIdx.x 写入，可处理类别数不能超过 blockDim。
    // blockDim == CNN_BLOCK(256)，且 validateNetSpec 已限 classes <= MAX_CLASSES(16) < 256，
    // 运行期恒有 cls == N.classes（默认 10，逐位不变）。若调用方绕过校验传入
    // classes > 256，此处截断到 blockDim，保证 softmax 绝不读取未初始化的 logits 槽位。
    const int cls = (N.classes < CNN_BLOCK) ? N.classes : CNN_BLOCK;

    if (threadIdx.x < cls) {
        const int o = threadIdx.x;
        const float* wrow = W + o * N.flat;
        float sum = bfc[o];
        for (int i = 0; i < N.flat; ++i)
            sum += wrow[i] * p2b[i];
        lo[o] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = lo[0];
        for (int i = 1; i < cls; ++i)
            if (lo[i] > mx) mx = lo[i];
        float s = 0.0f;
        for (int i = 0; i < cls; ++i) {
            const float e = expf(lo[i] - mx);
            pr[i] = e;
            s += e;
        }
        for (int i = 0; i < cls; ++i)
            pr[i] /= s;
    }
}

// ================= kloss_acc：GPU 端 loss / 准确率累加（B1） =================
// 每个 block 负责一个样本（b = blockIdx.x，grid = n）：先做 argmax 归约，
// 仅线程 0 用 atomicAdd 把 loss / correct / predCount 累加到全局缓冲。
// 语义与 CPU 串行一致：loss = -log(prob[label]+1e-8)；argmax 严格大于，
// 相等时保留较小类别下标（与 host 的首次出现即最大值一致）。
// 真值直接用 d_label（kgather 已写入本 batch 标签，等价 labels[order[start+s]]）。
__global__ void kloss_acc(const float* __restrict__ prob,
                          const int*   __restrict__ label,
                          int n,
                          float* __restrict__ d_lossSum,
                          int*   __restrict__ d_correct,
                          int*   __restrict__ d_predCount,
                          const NetSpec N)
{
    const int b = blockIdx.x;
    if (b >= n) return;

    __shared__ float sVal[CNN_BLOCK];
    __shared__ int   sIdx[CNN_BLOCK];

    const float* pr = prob + (size_t)b * N.classes;
    const int t = threadIdx.x;

    // 每线程至多负责一个类别（classes < blockDim=256）；
    // 非参线程置 -1（概率恒 >=0），保证不会赢得 argmax。
    if (t < N.classes) {
        sVal[t] = pr[t];
        sIdx[t] = t;
    } else {
        sVal[t] = -1.0f;
        sIdx[t] = t;
    }
    __syncthreads();

    // 树形归约求 argmax：严格大于；相等时保留较小类别下标
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            const float a  = sVal[t];
            const float c  = sVal[t + stride];
            const int   ia = sIdx[t];
            const int   ic = sIdx[t + stride];
            if (c > a || (c == a && ic < ia)) {
                sVal[t] = c;
                sIdx[t] = ic;
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        const int   pred = sIdx[0];
        const int   lab  = label[b];
        const float loss = -logf(pr[lab] + CNN_LOSS_EPS);
        atomicAdd(d_lossSum, loss);
        atomicAdd(d_correct, (pred == lab) ? 1 : 0);
        atomicAdd(&d_predCount[pred], 1);
    }
}

// ================= kb_fc：dlogits -> dW_fc / db_fc / dflat =================
__global__ void kb_fc(const float* __restrict__ prob,
                      const int*   __restrict__ label,
                      const float* __restrict__ p2,
                      const float* __restrict__ params,
                      float* __restrict__ grad,
                      float* __restrict__ dflat,
                      const NetSpec N)
{
    const int b = blockIdx.x;
    // H3：dlog 使用 net_spec.h 的固定上限 MAX_CLASSES 作为静态共享内存。
    // 编译期 static_assert(MAX_CLASSES>=16) 保证上限足够；运行期 validateNetSpec
    // 已把 classes 限制在 [2,16]。若调用方绕过校验，下面 cls 将访问钳制到
    // [0, MAX_CLASSES)，避免 dlog[MAX_CLASSES] 越界（防御性冗余，默认路径逐位不变）。
    __shared__ float dlog[MAX_CLASSES];

    const float* pr = prob + (size_t)b * N.classes;
    const float* p2b = p2 + (size_t)b * N.flat;
    float* dfl = dflat + (size_t)b * N.flat;
    const float* W = params + N.offFCW;
    const int lab = label[b];

    const int cls = (N.classes < MAX_CLASSES) ? N.classes : MAX_CLASSES;

    if (threadIdx.x < cls) {
        const int o = threadIdx.x;
        dlog[o] = pr[o] - ((o == lab) ? 1.0f : 0.0f);
    }
    __syncthreads();

    // db_fc
    if (threadIdx.x < cls)
        atomicAdd(&grad[N.offFCB + threadIdx.x], dlog[threadIdx.x]);

    // dW_fc（classes*flat；M3：以 size_t 计算总量，避免大 flat 时 int 溢出）
    for (size_t idx = (size_t)threadIdx.x; idx < (size_t)cls * (size_t)N.flat; idx += blockDim.x) {
        const int o = (int)(idx / (size_t)N.flat);
        const int i = (int)(idx - (size_t)o * (size_t)N.flat);
        atomicAdd(&grad[N.offFCW + o * N.flat + i], dlog[o] * p2b[i]);
    }

    // dflat[i] = sum_o W[o][i] * dlog[o]（样本私有，无竞争）
    for (int i = threadIdx.x; i < N.flat; i += blockDim.x) {
        float s = 0.0f;
        for (int o = 0; o < cls; ++o)
            s += W[o * N.flat + i] * dlog[o];
        dfl[i] = s;
    }
}

// ================= kb_pool_relu：第 li 层 [可选] Pool 反向 + ReLU 反向 -> dc =================
// pool 窗口互不重叠（validate 保证池化前维度为偶数），显式清零 4 个窗口位后写 max。
// L.pool==0 时退化为纯 ReLU 反向（dc = preRelu>0 ? dpool : 0）。
// 与改造前逐位一致：非 BN 时 preRelu==d_c[li]；调用方对输出整段 memset 亦相容。
__global__ void kb_pool_relu(const float* __restrict__ dpool,
                             const float* __restrict__ r,
                             const float* __restrict__ preRelu,
                             float* __restrict__ dc,
                             int li,
                             const NetSpec N)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    const int actSize = L.outC * L.H * L.W;
    const int pH = L.pool ? (L.H / 2) : L.H;
    const int pW = L.pool ? (L.W / 2) : L.W;
    const int poolSize = L.outC * pH * pW;
    const float* dp = dpool + (size_t)b * poolSize;
    const float* rb = r + (size_t)b * actSize;
    const float* pre = preRelu + (size_t)b * actSize;   // ReLU 输入：非 BN=d_c[li]，BN=d_b[li]
    float* dcb = dc + (size_t)b * actSize;

    if (!L.pool) {
        // pool=0：ReLU 反向，dc = pre>0 ? dpool : 0（dpool 与 dc 同形）
        for (int idx = threadIdx.x; idx < actSize; idx += blockDim.x)
            dcb[idx] = (pre[idx] > 0.0f) ? dp[idx] : 0.0f;
        return;
    }

    for (int idx = threadIdx.x; idx < poolSize; idx += blockDim.x) {
        const int c = idx / (pH * pW);
        const int rem = idx - c * (pH * pW);
        const int oh = rem / pW;
        const int ow = rem - oh * pW;
        const int ih = oh * 2, iw = ow * 2;

        const float* rr = rb + c * (L.H * L.W);
        float m = rr[ih * L.W + iw];
        int mh = ih, mw = iw;
        if (rr[ih * L.W + iw + 1] > m) { m = rr[ih * L.W + iw + 1]; mh = ih; mw = iw + 1; }
        if (rr[(ih + 1) * L.W + iw] > m) { m = rr[(ih + 1) * L.W + iw]; mh = ih + 1; mw = iw; }
        if (rr[(ih + 1) * L.W + iw + 1] > m) { m = rr[(ih + 1) * L.W + iw + 1]; mh = ih + 1; mw = iw + 1; }

        float* dch = dcb + c * (L.H * L.W);
        const float* prech = pre + c * (L.H * L.W);
        const int base = ih * L.W + iw;
        dch[base] = 0.0f;
        dch[base + 1] = 0.0f;
        dch[base + L.W] = 0.0f;
        dch[base + L.W + 1] = 0.0f;

        const int mi = mh * L.W + mw;
        // ReLU 反向：preRelu > 0 才回传（非 BN 时 preRelu==d_c[li]，逐位不变）
        dch[mi] = (prech[mi] > 0.0f) ? dp[idx] : 0.0f;
    }
}

// ================= kb_conv：第 li 层 Conv 反向 -> dK[li] / dB[li] / d(输入) =================
// li==0 无需 dinput（dpIn 传 nullptr）；li>0 把输入梯度写入 dpIn（上一层池化输出梯度）。
// addMode==0：覆盖写；addMode==1：累加写（残差网络分流）。addMode==0 与改造前逐位一致。
__global__ void kb_conv(const float* __restrict__ input,
                        const float* __restrict__ dc,
                        const float* __restrict__ params,
                        float* __restrict__ grad,
                        float* __restrict__ dpIn,
                        int li,
                        const NetSpec N,
                        int addMode)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    const int iH = netLayerInH(N, li);
    const int iW = netLayerInW(N, li);
    const int inSize  = L.inC * iH * iW;
    const int outSize = L.outC * L.H * L.W;
    const float* inb = input + (size_t)b * inSize;
    const float* dcb = dc + (size_t)b * outSize;
    const float* K = params + N.offK[li];

    // dB
    for (int oc = threadIdx.x; oc < L.outC; oc += blockDim.x) {
        const float* dcoc = dcb + oc * (L.H * L.W);
        float s = 0.0f;
        for (int k = 0; k < L.H * L.W; ++k) s += dcoc[k];
        atomicAdd(&grad[N.offB[li] + oc], s);
    }

    // dK（outC*inC*kH*kW）
    const int kArea = L.kH * L.kW;
    const int kStride = L.inC * kArea;
    for (int idx = threadIdx.x; idx < L.outC * kStride; idx += blockDim.x) {
        const int oc = idx / kStride;
        const int rem = idx - oc * kStride;
        const int ic = rem / kArea;
        const int k = rem - ic * kArea;
        const int kh = k / L.kW, kw = k - kh * L.kW;

        const float* dcoc = dcb + oc * (L.H * L.W);
        const float* inch = inb + ic * (iH * iW);
        float s = 0.0f;
        for (int oh = 0; oh < L.H; ++oh) {
            const int ih = oh * L.stride + kh - L.pad;
            if (ih < 0 || ih >= iH) continue;
            for (int ow = 0; ow < L.W; ++ow) {
                const int iw = ow * L.stride + kw - L.pad;
                if (iw < 0 || iw >= iW) continue;
                s += dcoc[oh * L.W + ow] * inch[ih * iW + iw];
            }
        }
        atomicAdd(&grad[N.offK[li] + oc * kStride + ic * kArea + kh * L.kW + kw], s);
    }

    // d(输入)：样本私有，无竞争。li==0 不需要 dinput（dpIn==nullptr）。
    if (dpIn) {
        float* dp = dpIn + (size_t)b * inSize;
        for (int idx = threadIdx.x; idx < inSize; idx += blockDim.x) {
            const int ic = idx / (iH * iW);
            const int rem = idx - ic * (iH * iW);
            const int ih = rem / iW;
            const int iw = rem - ih * iW;

            float s = 0.0f;
            for (int oc = 0; oc < L.outC; ++oc) {
                const float* dcoc = dcb + oc * (L.H * L.W);
                const float* kbase = K + oc * kStride + ic * kArea;
                for (int kh = 0; kh < L.kH; ++kh) {
                    const int numh = ih + L.pad - kh;
                    if (numh < 0 || (numh % L.stride) != 0) continue;
                    const int oh = numh / L.stride;
                    if (oh >= L.H) continue;
                    for (int kw = 0; kw < L.kW; ++kw) {
                        const int numw = iw + L.pad - kw;
                        if (numw < 0 || (numw % L.stride) != 0) continue;
                        const int ow = numw / L.stride;
                        if (ow >= L.W) continue;
                        s += dcoc[oh * L.W + ow] * kbase[kh * L.kW + kw];
                    }
                }
            }
            dp[idx] = addMode ? (dp[idx] + s) : s;
        }
    }
}

// ================= kadd：残差加法 / 分流 =================
// dst[idx] += src[idx]（每样本一段，样本私有、无竞争）。用于前向合并 skip，
// 以及反向把第 i 层输出梯度分流到 skip 源层。
__global__ void kadd(const float* __restrict__ src,
                     float* __restrict__ dst,
                     int nElem)
{
    const int b = blockIdx.x;
    const float* s = src + (size_t)b * nElem;
    float* d = dst + (size_t)b * nElem;
    for (int i = threadIdx.x; i < nElem; i += blockDim.x)
        d[i] += s[i];
}

// ================= 阶段6：1×1 投影残差 kernel =================
// 目标张量每个样本的形状：resPos=0 用 L 的池化后尺寸；resPos=1 用卷积输出（ReLU 前）尺寸。
// 源层输出空间尺寸与目标一致（validateNetSpec 保证），故源按 srcC×tH×tW 索引。
__device__ inline void projDims(const NetSpec& N, int li, int& tH, int& tW, int& srcC) {
    const LayerSpec& L = N.layer[li];
    srcC = N.layer[L.resFrom].outC;
    if (L.resPos == 0) { tH = L.pool ? (L.H / 2) : L.H; tW = L.pool ? (L.W / 2) : L.W; }
    else               { tH = L.H; tW = L.W; }
}

__global__ void kproj_add(const float* __restrict__ src,
                          const float* __restrict__ params,
                          float* __restrict__ dst,
                          int li,
                          const NetSpec N)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    int tH, tW, srcC;
    projDims(N, li, tH, tW, srcC);
    const int tHW = tH * tW;
    const int outSz = L.outC * tHW;
    const float* sb = src + (size_t)b * (srcC * tHW);
    float* db = dst + (size_t)b * outSz;
    const float* PW = params + N.offPW[li];
    const float* PB = params + N.offPB[li];

    for (int idx = threadIdx.x; idx < outSz; idx += blockDim.x) {
        const int oc = idx / tHW;
        const int pos = idx - oc * tHW;
        float sum = PB[oc];
        const float* w = PW + oc * srcC;
        for (int ic = 0; ic < srcC; ++ic)
            sum += w[ic] * sb[ic * tHW + pos];
        db[idx] += sum;
    }
}

__global__ void kproj_bwd(const float* __restrict__ src,
                          const float* __restrict__ dy,
                          const float* __restrict__ params,
                          float* __restrict__ grad,
                          float* __restrict__ dsrcOut,
                          int li,
                          const NetSpec N,
                          int addMode)
{
    const int b = blockIdx.x;
    const LayerSpec& L = N.layer[li];
    int tH, tW, srcC;
    projDims(N, li, tH, tW, srcC);
    const int tHW = tH * tW;
    const int outSz = L.outC * tHW;
    const int srcSz = srcC * tHW;
    const float* sb = src + (size_t)b * srcSz;
    const float* dyb = dy + (size_t)b * outSz;
    float* dsb = dsrcOut ? (dsrcOut + (size_t)b * srcSz) : nullptr;
    const float* PW = params + N.offPW[li];

    // dPB
    for (int oc = threadIdx.x; oc < L.outC; oc += blockDim.x) {
        const float* dyoc = dyb + oc * tHW;
        float s = 0.0f;
        for (int k = 0; k < tHW; ++k) s += dyoc[k];
        atomicAdd(&grad[N.offPB[li] + oc], s);
    }

    // dPW（outC×srcC）
    for (int idx = threadIdx.x; idx < L.outC * srcC; idx += blockDim.x) {
        const int oc = idx / srcC;
        const int ic = idx - oc * srcC;
        const float* dyoc = dyb + oc * tHW;
        const float* sic = sb + ic * tHW;
        float s = 0.0f;
        for (int k = 0; k < tHW; ++k) s += dyoc[k] * sic[k];
        atomicAdd(&grad[N.offPW[li] + idx], s);
    }

    // d(src)：样本私有，无竞争。addMode==1 时累加（残差分流）。
    if (dsb) {
        for (int idx = threadIdx.x; idx < srcSz; idx += blockDim.x) {
            const int ic = idx / tHW;
            const int pos = idx - ic * tHW;
            float s = 0.0f;
            for (int oc = 0; oc < L.outC; ++oc)
                s += dyb[oc * tHW + pos] * PW[oc * srcC + ic];
            dsb[idx] = addMode ? (dsb[idx] + s) : s;
        }
    }
}

// ================= kb_update：SGD with momentum + 梯度清零 =================
// 逐参数递归，不依赖 batch 统计：batch=1 时与 CPU 逐样本 SGD 完全等价。
// 首步 velocity 初值为 0，故 v = g，避免起步偏小（无需特殊分支）。
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
                          const NetSpec N)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N.total) return;

    // BN 的 running mean/var 是跨 batch 统计量、非训练参数：不更新、并清零其梯度位。
    // 其余参数（含 BN 的 γ/β）照常按区间 lr 更新。
    if (!isTrainableParam(N, p)) { grad[p] = 0.0f; return; }

    // [offK[0], offK[1]) 第 0 层(K0,B0,BN0)；[offK[1], offFCW) 第 1 层(K1,B1,BN1)；[offFCW,total) fc
    // 本阶段仍 2 层，故 lr 分组边界为 offK[1] 与 offFCW。
    const float lr = (p < N.offK[1]) ? lr_conv1
                   : (p < N.offFCW) ? lr_conv2
                   : lr_fc;

    // grad 是整个 batch 累加的梯度，须除以 batch 大小得到平均梯度（batch=1 时除以 1，与逐样本 SGD 等价）
    const float invBatch = 1.0f / (float)batch;
    float g = grad[p] * invBatch;

    // L2 weight decay：仅乘法权重（K/PW/FCW）叠加 wd*param；wd==0 时不进入（位保护，逐位不变）。
    if (wd != 0.0f && isWeightDecayParam(N, p)) g += wd * params[p];

    // momentum 递归：beta=0 时 v=g，step=v=g，退化为朴素 SGD
    const float v = beta * velocity[p] + g;
    velocity[p] = v;

    // Nesterov：look-ahead 梯度 g + beta*v（beta=0 时亦退化为 g）
    const float step = nesterov ? (g + beta * v) : v;
    params[p] -= lr * step;

    grad[p] = 0.0f;
}
