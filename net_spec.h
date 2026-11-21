#pragma once
#include <string>
#include <climits>

// 全项目共享的类别数上限（gpu_kernels.cuh 等引用本常量，避免各自 #define 漂移）
static constexpr int MAX_CLASSES = 16;

// 阶段1：网络最大层数（POD 数组上界，device 端可用）。
// 阶段2：数组上界即实际允许的最大深度（validateNetSpec 放开到 [1, MAX_DEPTH]）。
// 阶段6：8 -> 24，支撑 WRN/深 ResNet。NetSpec 体积仍 ~2.4KB < CUDA 4KB kernel 参数上限，
//         故 kernel 继续按值传 NetSpec 是安全的（见文件末 static_assert 记录）。
static constexpr int MAX_DEPTH = 24;

// 单层卷积规格（POD，主机与 device 共用，不能含 STL）。
//   outC/inC : 该层输出/输入通道
//   kH/kW    : 卷积核高/宽
//   H/W      : 该层卷积输出空间尺寸（池化前）
//   bn       : 该层是否启用 BatchNorm（0/1）
//   stride   : 卷积步长（>=1）
//   pad      : 卷积零填充（>=0）
//   pool     : 该层是否做 2×2 最大池化（2=是，0=否）
//   res      : 残差类型（阶段3）0=无、1=identity、2=1×1 投影（阶段6 放开）
//   resFrom  : 残差源层索引（0-based，须 < 本层 i；res==0 时忽略，约定为 -1）
//   resPos   : 残差相加位置（阶段6）0=本层最终输出（Pool 之后，旧语义，默认）；
//              1=pre-activation：本层 BN 之后、ReLU 之前（标准 basic block 位置）
// 残差语义（resPos=0，旧）：y = pool(relu(bn(conv(x)))) + skip，
//   identity 的 skip = layer[resFrom] 的最终输出；投影的 skip = Wproj * layer[resFrom].out。
// 残差语义（resPos=1，新）：b = bn(conv(x))；h = b + skip；y = pool(relu(h))。
// 形状约束（validateNetSpec 校验；1×1 投影 stride=1/pad=0，空间尺寸不变）：
//   * resPos=0：源层最终输出与目标层最终输出空间尺寸一致；identity 还需通道一致。
//   * resPos=1：源层最终输出与目标层卷积输出（Pool 前）空间尺寸一致；identity 还需通道一致。
// 默认 (stride,pad,pool,res,resFrom,resPos)=(1,0,2,0,-1,0) 即阶段1/2 行为，逐位不变。
struct LayerSpec {
    int outC = 0, inC = 0, kH = 0, kW = 0;
    int H = 0, W = 0;
    int bn = 0;
    int stride = 1, pad = 0, pool = 2;
    int res = 0, resFrom = -1, resPos = 0;
};

// 运行时网络规格（唯一定义；主机与 GPU 共用）。
// 阶段1：由「写死 2 层」改为「层列表 + 逐层偏移/数量」；对外文件格式不变。
// 参数扁平布局：K0|B0|[BN0g BN0b BN0mean BN0var]|K1|B1|[BN1g ...]|...|FCW|FCB
//   其中下标 i 为 0-based 层号；默认 depth=2 时与改造前逐位一致。
struct NetSpec {
    int inH = 32, inW = 32;
    int classes = 10;
    int depth = 2;                       // 网络深度（阶段2 起支持 [1, MAX_DEPTH]）
    LayerSpec layer[MAX_DEPTH] = {};     // 逐层规格

    // 逐层参数段偏移与数量
    int offK[MAX_DEPTH] = {}, nK[MAX_DEPTH] = {};
    int offB[MAX_DEPTH] = {}, nB[MAX_DEPTH] = {};
    // BN 的 γ/β/running_mean/running_var（bn=false 时数量全 0、偏移塌缩）
    // 注意：mean/var 属于跨 batch 统计量、非训练参数（见 isTrainableParam）
    int offBNg[MAX_DEPTH] = {},    nBNg[MAX_DEPTH] = {};
    int offBNb[MAX_DEPTH] = {},    nBNb[MAX_DEPTH] = {};
    int offBNmean[MAX_DEPTH] = {}, nBNmean[MAX_DEPTH] = {};
    int offBNvar[MAX_DEPTH] = {},  nBNvar[MAX_DEPTH] = {};

    // 阶段6：1×1 投影残差（res==2）的权重/偏置段，逐层。仅 res==2 的层非零，
    //   其它层数量为 0、偏移塌缩，故非投影网络的布局/total 逐位不变。
    //   nPW = outC * srcC（srcC = layer[resFrom].outC）；nPB = outC。
    int offPW[MAX_DEPTH] = {}, nPW[MAX_DEPTH] = {};
    int offPB[MAX_DEPTH] = {}, nPB[MAX_DEPTH] = {};

    // 全连接段与总量
    int flat = 0;
    int offFCW = 0, nFCW = 0, offFCB = 0, nFCB = 0, total = 0;

    // 是否启用 BatchNorm（全网络统一开关）
    bool bn = false;
};

// 由 layer[0..depth-1]（须已填好 outC/inC/kH/kW/bn/stride/pad/pool）逐层推导：
//   卷积输出尺寸 Hout=(Hin+2*pad-kH)/stride+1（整数）；pool=2 时再 Hout/=2，
//   pool=0 时直接作为下一层输入。每层参数数量与扁平布局偏移；末层输出即 flat；FC 段与 total。
// 布局顺序：K0|B0|[BN0]|K1|B1|[BN1]|...|FCW|FCB。
// bn=false 时各 BN 数量为 0，所有 BN 偏移塌缩，offK/offFCW/total 与改造前逐位一致
// （默认 32/64 下 total==42378）。
inline void computeNetSpec(NetSpec& s) {
    int iH = s.inH, iW = s.inW;
    int off = 0;
    for (int i = 0; i < s.depth && i < MAX_DEPTH; ++i) {
        LayerSpec& L = s.layer[i];
        if (L.stride < 1) L.stride = 1;   // 防御：非法步长按 1 处理（validate 会另行拒绝）
        // 卷积输出（整数除法；负数/0 由 validateNetSpec 拒绝）
        L.H = (iH + 2 * L.pad - L.kH) / L.stride + 1;
        L.W = (iW + 2 * L.pad - L.kW) / L.stride + 1;

        s.nK[i] = L.outC * L.inC * L.kH * L.kW;
        s.offK[i] = off; off += s.nK[i];
        s.nB[i] = L.outC;
        s.offB[i] = off; off += s.nB[i];

        const int c = s.bn ? L.outC : 0;
        s.nBNg[i] = s.nBNb[i] = s.nBNmean[i] = s.nBNvar[i] = c;
        s.offBNg[i]    = off; off += s.nBNg[i];
        s.offBNb[i]    = off; off += s.nBNb[i];
        s.offBNmean[i] = off; off += s.nBNmean[i];
        s.offBNvar[i]  = off; off += s.nBNvar[i];

        // 阶段6：res==2 的 1×1 投影段（放在该层 BN 段之后，不影响非投影层布局）。
        if (L.res == 2 && L.resFrom >= 0 && L.resFrom < i) {
            const int srcC = s.layer[L.resFrom].outC;
            s.nPW[i] = L.outC * srcC;
            s.offPW[i] = off; off += s.nPW[i];
            s.nPB[i] = L.outC;
            s.offPB[i] = off; off += s.nPB[i];
        } else {
            s.nPW[i] = s.nPB[i] = 0;
            s.offPW[i] = s.offPB[i] = off;
        }

        // 池化（2×2）输出作为下一层输入；pool=0 时卷积输出直接作为下一层输入
        iH = L.H; iW = L.W;
        if (L.pool) { iH /= 2; iW /= 2; }
    }
    const int lastC = (s.depth > 0 && s.depth <= MAX_DEPTH) ? s.layer[s.depth - 1].outC : 0;
    s.flat = lastC * iH * iW;
    s.nFCW = s.classes * s.flat;
    s.nFCB = s.classes;
    s.offFCW = off; off += s.nFCW;
    s.offFCB = off; off += s.nFCB;
    s.total = off;
}

// 阶段2：默认「同构 conv→pool 栈」规格生成（供 gen/bkc 新建模型用）。
//   depth==2：layer0 k=5/pad0、layer1 k=3/pad0（与阶段1 完全一致，逐位不变）；
//   depth>2 ：每层 k=3/pad1/stride1（保持尺寸），pool=2 每层减半（32→16→8→…），
//             通道 layer0=c1、其余=c2，避免空间立即塌缩。
// 如需更深的网络可手写 P<i> 覆盖（见文件格式说明）。
inline NetSpec makeDefaultNetSpec(int inH, int inW, int classes, bool bn, int depth,
                                  int c1, int c2) {
    NetSpec s;
    s.inH = inH; s.inW = inW; s.classes = classes;
    s.bn = bn; s.depth = depth;
    for (int i = 0; i < depth && i < MAX_DEPTH; ++i) {
        LayerSpec& L = s.layer[i];
        L.outC = (i == 0) ? c1 : c2;
        L.inC = (i == 0) ? 1 : s.layer[i - 1].outC;
        if (depth == 2) {
            L.kH = L.kW = (i == 0) ? 5 : 3;
            L.pad = 0;
        } else {
            L.kH = L.kW = 3;
            L.pad = 1;
        }
        L.stride = 1;
        L.pool = 2;
        L.bn = bn ? 1 : 0;
    }
    computeNetSpec(s);
    return s;
}

// 阶段3：默认「残差网络」规格（gen/bkc --res 用）。
//   * layer0 为 stem：k=3/pad=1/stride=1/pool=2，把空间减半；
//   * layer1..depth-1 全部 pool=0（保持尺寸），使 block 内形状一致；
//   * 所有层同通道 c，末层带 identity 残差指向 layer0（block 输入），
//     即 y = pool(relu(conv(x))) + stem_out，可表达 ResNet/WRN basic block。
//   depth>=2；depth==2 时退化为「stem + 一个 conv」，同样合法。
//   flat = c * (inH/2) * (inW/2)（pool 只发生在 stem）。
inline NetSpec makeResNetSpec(int inH, int inW, int classes, bool bn, int depth, int c) {
    NetSpec s;
    s.inH = inH; s.inW = inW; s.classes = classes;
    s.bn = bn; s.depth = depth;
    for (int i = 0; i < depth && i < MAX_DEPTH; ++i) {
        LayerSpec& L = s.layer[i];
        L.outC = c;
        L.inC = (i == 0) ? 1 : c;
        L.kH = L.kW = 3;
        L.pad = 1;
        L.stride = 1;
        L.pool = (i == 0) ? 2 : 0;   // 仅 stem 池化，block 内保持空间尺寸
        L.bn = bn ? 1 : 0;
        L.res = 0; L.resFrom = -1;
    }
    if (depth >= 2) {
        LayerSpec& last = s.layer[depth - 1];
        last.res = 1;
        last.resFrom = 0;            // 末层残差指向 stem 输出（block 输入）
    }
    computeNetSpec(s);
    return s;
}

// 阶段6：1×1 投影残差网络（gen/bkc --res2 用）。
//   与 makeResNetSpec 相同的 stem+block 结构，区别仅在末层残差为 res=2（1×1 投影）：
//   y = pool(relu(conv(x))) + Wproj * stem_out，Wproj 为 c×c 的 1×1 卷积（含偏置）。
//   用于通道对齐（本构造同通道，1×1 初始非恒等）；validateNetSpec 放开 res=2。
inline NetSpec makeProjResNetSpec(int inH, int inW, int classes, bool bn, int depth, int c) {
    NetSpec s = makeResNetSpec(inH, inW, classes, bn, depth, c);
    if (depth >= 2) {
        LayerSpec& last = s.layer[depth - 1];
        last.res = 2;
        last.resFrom = 0;
        last.resPos = 0;
    }
    computeNetSpec(s);
    return s;
}

// 阶段6：pre-activation 残差网络（gen/bkc --preact 用）。
//   与 makeResNetSpec 相同结构，区别在末层残差加在「BN 之后、ReLU 之前」（resPos=1）：
//   b = bn(conv(x))；h = b + stem_out；y = pool(relu(h))。默认非 BN 时 b=conv 输出。
inline NetSpec makePreactResNetSpec(int inH, int inW, int classes, bool bn, int depth, int c) {
    NetSpec s = makeResNetSpec(inH, inW, classes, bn, depth, c);
    if (depth >= 2) {
        LayerSpec& last = s.layer[depth - 1];
        last.res = 1;
        last.resFrom = 0;
        last.resPos = 1;
    }
    computeNetSpec(s);
    return s;
}

// 使该函数在 CUDA 编译单元中同时可被 device 代码（如 kb_update kernel）调用；
// 纯 C++（main.cpp）编译时 __CUDACC__ 未定义，宏为空，保持原 host 行为不变。
#ifdef __CUDACC__
#define NETSPEC_HOST_DEVICE __host__ __device__
#else
#define NETSPEC_HOST_DEVICE
#endif

// BN 的 running_mean/running_var 属于统计量、非训练参数：优化器不应更新它们。
// p 落在任一层的 BN mean/var 区间时返回 false；其余（含所有非 BN 参数）返回 true。
NETSPEC_HOST_DEVICE inline bool isTrainableParam(const NetSpec& s, int p) {
    for (int i = 0; i < s.depth && i < MAX_DEPTH; ++i) {
        if (s.nBNmean[i] > 0 && p >= s.offBNmean[i] && p < s.offBNmean[i] + s.nBNmean[i]) return false;
        if (s.nBNvar[i]  > 0 && p >= s.offBNvar[i]  && p < s.offBNvar[i]  + s.nBNvar[i])  return false;
    }
    return true;
}

// L2 weight decay 仅作用于乘法权重：卷积核 K 段、1×1 投影 PW 段、FC 权重 FCW 段。
// 用 NetSpec 的段偏移（offK/nK、offPW/nPW、offFCW/nFCW）判断。
// BN 的 γ/β、所有 bias（B/PB/FCB）、统计量 mean/var 一律不衰减。
NETSPEC_HOST_DEVICE inline bool isWeightDecayParam(const NetSpec& s, int p) {
    for (int i = 0; i < s.depth && i < MAX_DEPTH; ++i) {
        if (s.nK[i]  > 0 && p >= s.offK[i]  && p < s.offK[i]  + s.nK[i])  return true;
        if (s.nPW[i] > 0 && p >= s.offPW[i] && p < s.offPW[i] + s.nPW[i]) return true;
    }
    if (s.nFCW > 0 && p >= s.offFCW && p < s.offFCW + s.nFCW) return true;
    return false;
}

// 第 li 层的输入空间尺寸（主机/device 共用）：
//   li==0 用网络输入 inH/inW；否则用上一层 ReLU 输出（pool 时 /2，pool=0 时原尺寸）。
NETSPEC_HOST_DEVICE inline int netLayerInH(const NetSpec& s, int li) {
    if (li <= 0) return s.inH;
    const LayerSpec& P = s.layer[li - 1];
    return P.pool ? (P.H / 2) : P.H;
}
NETSPEC_HOST_DEVICE inline int netLayerInW(const NetSpec& s, int li) {
    if (li <= 0) return s.inW;
    const LayerSpec& P = s.layer[li - 1];
    return P.pool ? (P.W / 2) : P.W;
}

// 逐层校验规格自洽；阶段2 起 depth ∈ [1, MAX_DEPTH]。
inline bool validateNetSpec(const NetSpec& s, std::string* err = nullptr) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };

    // 0) 深度范围
    if (s.depth < 1 || s.depth > MAX_DEPTH) return fail("depth 越界");

    // 1) 基本尺寸
    if (s.inH < 3 || s.inW < 3) return fail("输入尺寸必须 >= 3");

    // 2) 类别数上下限（集中守卫，防止 FC 维/GPU 共享内存越界）
    if (s.classes < 2) return fail("classes 必须 >= 2");
    if (s.classes > MAX_CLASSES) return fail("classes 超出 MAX_CLASSES(16)");

    // 3) 逐层校验：通道/核/步长/填充/池化合法性、层间通道连续、卷积输出尺寸、池化前偶数
    int prevOut = 1;                                  // 第 0 层输入恒为 1 通道
    int iH = s.inH, iW = s.inW;                       // 逐层累积的输入尺寸
    int poolOutH[MAX_DEPTH] = {}, poolOutW[MAX_DEPTH] = {};  // 每层 Pool 后形状（残差源比对用）
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        if (L.outC < 1 || L.inC < 1) return fail("通道数必须 >= 1");
        if (L.kH < 1 || L.kW < 1) return fail("卷积核必须 >= 1");
        if (L.stride < 1) return fail("stride 必须 >= 1");
        if (L.pad < 0) return fail("pad 必须 >= 0");
        if (L.pool != 0 && L.pool != 2) return fail("pool 只能为 0 或 2");
        if (i > 0 && L.inC != prevOut) return fail("层间输入/输出通道不连续");

        // 卷积输出：先判分子非负，避免负号整除截断掩盖非法规格
        const long long numH = (long long)iH + 2LL * L.pad - L.kH;
        const long long numW = (long long)iW + 2LL * L.pad - L.kW;
        if (numH < 0 || numW < 0) return fail("卷积输出维度过小（核过大）");
        const long long oH = numH / L.stride + 1;
        const long long oW = numW / L.stride + 1;
        if (oH < 1 || oW < 1) return fail("卷积输出维度过小（核过大）");
        if (L.H != oH || L.W != oW) return fail("卷积输出维度与推导不一致");

        iH = (int)oH; iW = (int)oW;
        if (L.pool) {
            if (iH < 2 || iW < 2 || (iH % 2) || (iW % 2))
                return fail("池化前维度必须为偶数");
            iH /= 2; iW /= 2;
        }
        poolOutH[i] = iH; poolOutW[i] = iW;

        // 阶段3/6：残差校验（仅 res!=0 的层进入，非残差路径零开销、行为逐位不变）。
        //   res=1 identity；res=2 为 1×1 投影（通道可不同，空间须一致）。
        //   resPos=0 加在本层最终输出（Pool 后）；resPos=1 加在 BN 后、ReLU 前（卷积输出形状）。
        if (L.res != 0) {
            if (L.res != 1 && L.res != 2)
                return fail("残差类型仅支持 0（无）、1（identity）、2（1×1 投影）");
            if (L.resPos != 0 && L.resPos != 1)
                return fail("残差位置 resPos 仅支持 0（Pool 后）或 1（pre-activation）");
            if (L.resFrom < 0 || L.resFrom >= i)
                return fail("残差源层索引必须满足 0 <= resFrom < 层号");
            const LayerSpec& S = s.layer[L.resFrom];
            // 目标形状：resPos=0 用池化后；resPos=1 用卷积输出（ReLU 前）
            const int tgtH = (L.resPos == 0) ? iH : (int)oH;
            const int tgtW = (L.resPos == 0) ? iW : (int)oW;
            if (poolOutH[L.resFrom] != tgtH || poolOutW[L.resFrom] != tgtW)
                return fail("残差要求源层输出与本层目标空间尺寸一致");
            if (L.res == 1 && S.outC != L.outC)
                return fail("identity 残差要求源层与本层通道一致（同 C）");
        }
        prevOut = L.outC;
    }

    // 4) flat（末层输出：pool 为池化后、否则为卷积输出）用 long long 防溢出
    long long flatExp = (long long)prevOut * iH * iW;
    if (flatExp <= 0 || flatExp > (long long)INT_MAX)
        return fail("flat 计算溢出（末层输出超出 int）");
    if (s.flat != flatExp) return fail("flat 不一致");

    // 5) 逐层重算参数段数量/偏移，验证与结构体字段一致（long long 防累加溢出）
    long long off = 0;
    long long totalExp = 0;
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        long long nK = (long long)L.outC * L.inC * L.kH * L.kW;
        long long nB = L.outC;
        if (nK <= 0 || nK > (long long)INT_MAX) return fail("nK 溢出或非正");
        if (nB <= 0 || nB > (long long)INT_MAX) return fail("nB 溢出或非正");
        if (s.nK[i] != nK || s.nB[i] != nB) return fail("参数段数量与推导值不一致");
        if (s.offK[i] != off) return fail("K 段偏移不自洽"); off += nK;
        if (s.offB[i] != off) return fail("B 段偏移不自洽"); off += nB;

        if (s.bn) {
            if (s.nBNg[i] != L.outC || s.nBNb[i] != L.outC ||
                s.nBNmean[i] != L.outC || s.nBNvar[i] != L.outC)
                return fail("BN 数量与通道不一致");
        } else {
            if (s.nBNg[i] || s.nBNb[i] || s.nBNmean[i] || s.nBNvar[i])
                return fail("非 BN 时 BN 数量应为 0");
        }
        if (s.offBNg[i]    != off) return fail("BNγ 偏移不自洽"); off += s.nBNg[i];
        if (s.offBNb[i]    != off) return fail("BNβ 偏移不自洽"); off += s.nBNb[i];
        if (s.offBNmean[i] != off) return fail("BN mean 偏移不自洽"); off += s.nBNmean[i];
        if (s.offBNvar[i]  != off) return fail("BN var 偏移不自洽");  off += s.nBNvar[i];

        // 阶段6：1×1 投影段数量/偏移自洽（仅 res==2 非零；其它为 0 且偏移塌缩）。
        if (L.res == 2) {
            if (L.resFrom < 0 || L.resFrom >= i) return fail("投影残差源层索引非法");
            const long long nPW = (long long)L.outC * s.layer[L.resFrom].outC;
            const long long nPB = L.outC;
            if (nPW <= 0 || nPW > (long long)INT_MAX) return fail("nPW 溢出或非正");
            if (s.nPW[i] != nPW || s.nPB[i] != nPB) return fail("投影段数量与推导值不一致");
        } else {
            if (s.nPW[i] || s.nPB[i]) return fail("非投影层投影段数量应为 0");
        }
        if (s.offPW[i] != off) return fail("PW 偏移不自洽"); off += s.nPW[i];
        if (s.offPB[i] != off) return fail("PB 偏移不自洽"); off += s.nPB[i];
    }

    // 6) FC 段与 total
    long long nFCW = (long long)s.classes * s.flat;
    long long nFCB = s.classes;
    if (nFCW <= 0 || nFCW > (long long)INT_MAX) return fail("nFCW 溢出或非正");
    if (nFCB <= 0 || nFCB > (long long)INT_MAX) return fail("nFCB 溢出或非正");
    if (s.nFCW != nFCW || s.nFCB != nFCB) return fail("参数段数量与推导值不一致");
    if (s.offFCW != off) return fail("FCW 偏移不自洽"); off += nFCW;
    if (s.offFCB != off) return fail("FCB 偏移不自洽"); off += nFCB;
    totalExp = off;
    if (totalExp <= 0 || totalExp > (long long)INT_MAX) return fail("total 溢出或非正");
    if (s.total != totalExp) return fail("total 与各参数段之和不一致");

    return true;
}

// 阶段6：NetSpec 需按值传入 CUDA kernel，编译器对 kernel 参数有 4KB 上限。
// MAX_DEPTH 提升到 24 后确认 NetSpec 仍显著小于该上限，故保持按值传参不改签名。
static_assert(sizeof(NetSpec) < 4096,
              "NetSpec 超过 CUDA kernel 参数上限（4KB），需改为按 const NetSpec* 传参");
