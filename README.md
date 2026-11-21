# CNN2 工程说明（唯一权威文档）

> 本文件是 CNN2 项目的**唯一权威说明**。多人协作时请以本文件 + 代码为准；
> 若发现文档与代码不一致，**以代码为准**，并请顺手更新本文件。
>
> 项目的所有网络维度、参数偏移、BN 规则、命令行语义均已在源码中固化，
> 请勿凭记忆修改，改动前先读完对应章节。

---

## 目录

1. [项目概述](#1-项目概述)
2. [快速开始](#2-快速开始)
3. [网络架构与 NetSpec（运行时规格）](#3-网络架构与-netspec运行时规格)
4. [数据格式与预处理](#4-数据格式与预处理)
5. [数据增强](#5-数据增强)
6. [训练](#6-训练)
7. [BatchNorm（重点）](#7-batchnorm重点)
8. [模型文件格式](#8-模型文件格式)
9. [命令行参数总表](#9-命令行参数总表)
10. [测试 / 评估（集成 / 融合 / TTA / 判错）](#10-测试--评估集成--融合--tta--判错)
11. [模型签名 #SIG 与 --P](#11-模型签名-sig-与---p)
12. [构建](#12-构建)
13. [工程文件说明](#13-工程文件说明)
14. [部署脚本](#14-部署脚本)
15. [编码规范与陷阱](#15-编码规范与陷阱)
16. [附录：常量速查与常见错误](#16-附录常量速查与常见错误)

---

## 1. 项目概述

### 1.1 这是什么

CNN2 是一个**从零手写实现的手写数字识别（0–9）卷积神经网络**，不使用 PyTorch / TensorFlow
等任何深度学习框架。卷积、池化、ReLU、Softmax、全连接的前向与反向、SGD + Momentum
优化、可选 BatchNorm、数据增强、集成融合、TTA 等全部为自研 C++/CUDA 代码。

同一份代码既可以：

- 在 **GPU（CUDA）** 上做真 BatchNorm 训练与批量推理；
- 在 **CPU** 上做训练（不含 BN）与单图 / 批量推理；
- 通过 **Win32 GUI** 手写识别（CPU 推理，支持 BN 模型）。

### 1.2 技术栈

| 项目 | 说明 |
| --- | --- |
| IDE / 编译器 | Visual Studio 2022，MSVC PlatformToolset **v143**（实测 14.44） |
| CUDA | **CUDA 12.9**（工程内 `CUDA 12.9.props/targets`），`cudart` 动态链接 |
| C++ 标准 | `stdcpp17` |
| GUI | 纯 Win32（user32 / gdi32 / kernel32），无第三方库 |
| 目标 GPU | P100（`sm_60`）与 GTX 1070（`sm_61`），同时编入 `compute_60` PTX |
| 运行平台 | Windows x64 |

### 1.3 免安装部署

`deploy\` 目录即发布目录，把 **`CNN2.exe` 与同目录下 4 个 DLL 一起拷贝**即可运行，
目标机器**无需安装 CUDA Toolkit，也无需单独安装 VC++ 运行库**：

```
CNN2.exe
cudart64_12.dll        CUDA 运行时（共享链接所需）
msvcp140.dll           MSVC C++ 运行库
vcruntime140.dll       MSVC C 运行库
vcruntime140_1.dll     MSVC C 运行库（异常处理）
```

注意：**不能只拷 exe**。`KERNEL32.dll`、`api-ms-win-crt-*.dll` 是 Windows 系统自带，无需拷贝。

### 1.4 版本与兼容

CNN2 有**两个相互独立**的版本号，请勿混淆：

| 版本 | 取值 | 位置 | 含义 |
| --- | --- | --- | --- |
| **程序版本** | `1.0.0` | 源码 `version.h` 的 `CNN2_VERSION`；`CNN2.exe --version` 打印 | 本软件（`CNN2.exe`）的语义化版本，随代码里程碑递增 |
| **模型格式版本** | `2` | 模型文件 `#SIG` 行末尾的 `ver=` 字段；`version.h` 的 `CNN2_MODEL_VER` | 权重文件布局含义：`1`=旧两层格式；`2`=可变深度 / `P<i>` / 残差 |

- `--version` 只打印程序版本、模型格式版本与构建日期/时间，**不加载模型/数据、不训练**，
  不初始化 CUDA，打印后以退出码 **0** 结束。可出现在任意位置，优先于其它模式。
- **两者关系**：程序版本描述「哪一版可执行文件」；模型格式版本描述「权重文件怎么读」。
  程序通常能向后兼容读取更旧的模型格式，但旧程序不一定能读新格式。
- **升级 / 兼容策略**：
  - 仅新增/修改 CLI、修复实现、调整训练默认值等**不改文件布局**时，只递增程序版本，
    模型格式版本保持不变。
  - 一旦改动 `K<i>` / `B<i>` / `BN<i>` / `P<i>` / `R<i>` / `FC` 的布局或语义，
    **必须递增 `CNN2_MODEL_VER`**，在 `#SIG` 写新的 `ver=`，并在 `CHANGELOG.md`
    记录不兼容项。
  - `ver=1`（含无 `ver=` 字段、甚至无 `#SIG` 的最老文件）始终可被当前程序加载；
    `ver=2` 文件按 `P<i>` / `R<i>` 解析，缺失则回退默认值（`P<i>`=`1 0 2`、无残差）。
  - **模型文件有 acm 二进制（默认）与 txt 文本两种等价格式**（共用同一 `ver=` 语义、可无损互转），
    详见 [8.5](#85-acm-二进制格式默认--txtmodel-与后缀规则)。

---

## 2. 快速开始

以下示例假定 `CNN2.exe` 与 DLL 同目录，数据目录使用脚本里的默认路径
（见 [14. 部署脚本](#14-部署脚本)）。

```bat
rem 1) 新建（随机初始化）模型：默认 32/64 通道，输入 32x32，10 类
CNN2.exe dummy gen --model model.txt

rem 2) 训练（bkc）：若 model.txt 已存在则加载后继续训练，否则随机初始化
CNN2.exe C:\path\trainDigits bkc --gpu --epochs 50 --batch 256 --model model.txt

rem 3) 测试（默认 CPU，--gpu 开启批量 GPU 推理）
CNN2.exe C:\path\testDigits test --model model.txt
CNN2.exe C:\path\testDigits test --gpu --model model.txt --outerror

rem 4) 单图推理（第二个参数不是 gen/bkc/test 即按单图处理）
CNN2.exe C:\path\one.txt --model model.txt

rem 5) Win32 GUI（默认加载 model.txt）
CNN2.exe --gui
CNN2.exe --gui --model model.txt --fpng C:\path\one.txt

rem 6) 终端 TUI 指令台（叠加功能，默认仍是普通命令行）
CNN2.exe tui
```

> `gen` 的第一个参数（示例中的 `dummy`）会被解析但不使用；`bkc` 的第一个参数是**训练目录**。

> **模型格式提示**：自本版起默认保存为 **acm 二进制**（`--model` 省略时默认名 `model.acm`）。
> 上文示例沿用 `model.txt` 以便阅读；如需文本模型，显式加 `--txtmodel`（或把路径写成 `.txt`）。
> 详见 [8.5](#85-acm-二进制格式默认--txtmodel-与后缀规则)。

---

## 3. 网络架构与 NetSpec（运行时规格）

### 3.1 结构

```
Conv(k1×k1, 1→c1) → [BN] → ReLU → MaxPool2×2
    → Conv(k2×k2, c1→c2) → [BN] → ReLU → MaxPool2×2
    → Flatten → FC(flat→classes) → Softmax
```

`[BN]` 表示可选 BatchNorm（见第 7 节）。

### 3.2 默认维度与派生维度

默认输入 **32×32**、`c1=32`、`k1=5`、`c2=64`、`k2=3`、`classes=10`：

| 量 | 计算 | 默认值 |
| --- | --- | --- |
| Conv1 输出 | `c1H=inH-k1+1` | 32×28×28（c1Size=25088） |
| Pool1 输出 | `p1H=c1H/2` | 32×14×14（p1Size=6272） |
| Conv2 输出 | `c2H=p1H-k2+1` | 64×12×12（c2Size=9216） |
| Pool2 输出 | `p2H=c2H/2` | 64×6×6 |
| Flat | `flat=c2*p2H*p2W` | 2304 |
| FC | `classes×flat` | 10×2304 |

约束（`validateNetSpec`）：所有尺寸为正；`depth ∈ [1, MAX_DEPTH(24)]`；每层 `stride>=1`、
`pad>=0`、`pool ∈ {0,2}`；池化前维度必须为偶数。depth=2 默认参数与旧版逐位一致。

### 3.3 唯一规格来源：`net_spec.h` 的 `NetSpec`

**`NetSpec` 是整个工程的唯一规格来源**，同时被主机侧（`main.cpp`）与 GPU 侧
（`.cu` / `.cuh`）使用。它保存两类信息：

1. **维度**：`inH/inW`、`c1/k1`、`c1H/c1W`、`p1H/p1W`、`c2/k2`、`c2H/c2W`、
   `p2H/p2W`、`flat`、`classes`；
2. **参数扁平布局偏移**：每个参数段的起止偏移与数量。

`makeDefaultNetSpec(inH,inW,classes,bn,depth,c1,c2)` 生成默认规格，再由
`computeNetSpec` 从逐层 `outC/inC/kH/kW/stride/pad/pool` 推导全部派生量（尺寸与参数偏移）。

> **铁律**：参数偏移必须 CPU/GPU **使用同一份 `NetSpec`**。`flattenParams` / `scatterParams`
> 与所有 CUDA kernel 都从 `g_net` / 传入的 `spec` 读取偏移；**绝不允许**再次硬编码偏移，
> 否则会静默写错内存。

### 3.4 参数扁平布局

非 BN（默认 32/64 下 `total=42378`）：

```
[0,800)        K1   Conv1 权重 (oc,ic,kh,kw)，ic=1，32×5×5
[800,832)      B1   Conv1 偏置 (32)
[832,19264)    K2   Conv2 权重 (oc,ic,kh,kw)，64×32×3×3
[19264,19328)  B2   Conv2 偏置 (64)
[19328,42368)  FCW  全连接权重 (o,i) row-major，10×2304
[42368,42378)  FCB  全连接偏置 (10)
```

BN 模型在 B1 后插入 BN1 段、在 B2 后插入 BN2 段（各含 γ/β/mean/var），
`total=42762`（默认规模），详细偏移见 `net_spec.h` 中 `makeDefaultNetSpec` / `computeNetSpec` 的布局注释。

### 3.5 架构运行时可配置（`--c1` / `--c2` / `--depth`）

- `gen` 时用 `--c1` / `--c2` / `--depth`（默认 32/64/2）构造规格。
  - `depth=2`：核尺寸固定 `k1=5`、`k2=3`，与改造前逐位一致；
  - `depth>2`：使用默认 schedule，见 3.6。
- `bkc` / `test` / 单图 / GUI 时：**以模型文件为准**，由 `peekModelDims` 从权重文件
  反推每层的 `outC/inC/kH/kW/stride/pad/pool` 与 `classes`。若 `bkc` 时 `--model`
  已存在且与 `--c1/--c2` 冲突，会打印警告并**以文件为准**。
- GPU kernel 已支持任意规模与 `stride/pad/pool`，无需再拦截。

### 3.6 可变深度 / `P<i>` / `#SIG ver=2`（阶段2）

- 每层新增 `stride`/`pad`/`pool` 字段（`pool∈{0,2}`，0=不池化），默认 `(1,0,2)`。
  前向尺寸：`Hout=(Hin+2*pad-kH)/stride+1`；`pool=2` 时再 `Hout/=2`。
- **默认 schedule（depth>2）**：每层 `Conv(k=3, pad=1, stride=1)`（保持尺寸）+
  `pool=2`（每层减半 `32→16→8→…`），通道 `layer0=c1`、其余 `=c2`。
  该 schedule 在文件里以显式 `P<i>` 落盘。
- **`P<i> <stride> <pad> <pool>`**：写在每层 `B<i>`/`[BN<i>]` 之后；缺失按默认 `(1,0,2)`。
- **`#SIG ver=2`**：仅当 `depth>2` 或存在非默认 `P<i>` 时追加到签名行末尾
  （`... | depth=N | ver=2`）；depth=2 默认路径不写，保持旧格式。
- 旧 depth=2 文件（含无 `#SIG` 最老文件）读取路径不变；depth≥3 且不带任何新 token
  （仅多 K/B 段）的文件同样可加载（“读到 FC 为止”）。
- 加载时 `depth` 以**实际读到的层数**为准，避免 `#SIG` 旧值把深度静默降级。

---

## 4. 数据格式与预处理

### 4.1 目录与文件格式

- 训练 / 测试目录为**单层目录**，程序扫描 `目录\*.txt`（不递归子目录）。
- 每个 `.txt` = 一张图：**32 行 × 32 字符**，字符只允许 `'0'` 或 `'1'`（`1`=笔画，`0`=背景）。
- 行内长度不足 32 或出现非法字符会报错；空行被跳过。

### 4.2 标签

标签 = **文件名首字符**，必须为 `'0'`–`'9'`。例如 `3_a.txt` 标签为 3，`07_1.txt` 标签为 0。
首字符非数字的文件会被跳过。

### 4.3 预处理管线

总入口 `preprocessImage`，默认顺序：

```
cropToBBox → deskew → resizeToTarget → centerByCentroid → thicknessNormalize → finalizeBinary
```

- **crop**：按前景包围盒裁剪并平移到 (0,0)，无前景则跳过。
- **deskew**：基于二阶矩的剪切校正（轴 `axis=0`），角度限幅 `±deskewMaxDeg`，
  低各向异性（接近圆）或退化时跳过。
- **resize**：内容缩放到目标框最长边 `target`（默认保持长宽比；`--prep_stretch` 时
  拉伸为 `target×target`）。下采样用覆盖率阈值 0.5，上采样退化为最近邻。
- **center**：前景质心整数平移到画布中心。
- **thick**：按目标墨迹比迭代 3×3 膨胀/腐蚀，腐蚀后为空则回滚。
- **finalizeBinary**：总入口强制归一到 0/1。

**所有几何变换只用整数平移 / 最近邻 / 覆盖率阈值，不做灰度插值，输出严格二值**。

### 4.4 预处理参数与默认值

| 参数 | 默认 | 语义 |
| --- | --- | --- |
| 步骤开关 | crop=on, deskew=off, resize=on, center=on, thick=off | 见 `--prep` |
| `--prep_target N` | **28** | resize 目标框最长边，clamp 到 16–30 |
| `--prep_stretch` | off | 拉伸到 target×target（否则保持长宽比） |
| `--prep_ink X` | 0.20 | thick 目标墨迹比，`0`=关闭 |
| `--prep_deskew_max D` | 15 | deskew 角度限幅（度） |
| `--diag` | off | 打印预处理前后统计（墨迹比/质心/包围盒/二值断言），前 10 张限流 |

内部固定：`thickMode=0`（自动）、`thickIters=5`、`thickTol=0.01`。

> **重要**：训练与测试必须使用**完全相同**的预处理配置，否则识别精度会明显下降。
> 典型翻车场景：训练用默认 target=28，测试忘了加 `--prep_target 28`。

### 4.5 唯一带预处理的读图入口

`loadImagePreprocessed` 是训练、测试、单图、GUI 共用的唯一读图 + 预处理入口，
保证三方预处理一致（GUI 走 `preprocessImage` 后再前向）。

---

## 5. 数据增强

`--aug` 系列默认**全部关闭**。所有几何变换用「逆向映射 + 最近邻」，弹性形变用
「随机位移场 + 高斯平滑 + 最近邻重采样 + 重新二值化」（Simard 2003），**不做灰度插值**。

| 参数 | 默认 | 语义 |
| --- | --- | --- |
| `--aug` | off | 总开关 |
| `--aug_k K` | 2 | 离线扩增副本数（GPU 路径每张原图额外生成 K 张） |
| `--aug_shift X` | 3 | 随机平移幅度 ±X px |
| `--aug_rotate X` | 12 | 随机旋转幅度 ±X 度 |
| `--aug_scale X` | 0.1 | 缩放抖动半幅（0.1 ⇒ 0.9–1.1），内部 clamp 到 [0.5,2.0] |
| `--aug_elastic A` | 0（关） | 弹性形变强度 α，建议从 4 起 |
| `--aug_elastic_sigma S` | 4.5 | 弹性位移场高斯 σ，<0.5 时置 0.5 |
| `--aug_thick P` | 50 | 笔画粗细增强触发概率 %，0=关闭，clamp 到 0–100 |

单样本顺序：几何 → 弹性 → 粗细 → 二值化（`augmentRandom`）。

**训练路径差异**：

- **GPU**：离线扩增。加载阶段每张原图生成 `aug_k` 个增强副本，之后训练不再随机增强。
  每样本用 `(seedBase, i)` 派生独立 RNG，与线程划分无关 ⇒ 固定 `--seed` 可复现。
- **CPU**：在线增强。每个 epoch 每个样本训练前随机增强一次。

---

## 6. 训练

### 6.1 模式与后端选择

- 模式：`bkc`（训练）。
- 后端：`--cpu` 强制 CPU；`--gpu` 强制 GPU；都不给则自动（有 CUDA 设备走 GPU）。
- **CPU 不支持 BatchNorm**：`bkc --BN` 最终走 CPU 时会报错退出（见第 7 节）。

### 6.2 优化器：SGD + Momentum（可选 Nesterov）

- 交叉熵损失；`softmax → dlogits = prob - onehot`。
- 逐参数动量更新：`v = β·v + g`；Nesterov 时 `step = g + β·v`，否则 `step = v`；
  `w -= lr·step`。`β=0` 退化为朴素 SGD。
- 默认 `--momentum 0.9`（开启），`--nesterov` 默认关。
- 三组学习率独立：`--lr_fc`(1e-4)、`--lr_conv1`(1e-4)、`--lr_conv2`(5e-4)。
- 可用 `--no_conv1 / --no_conv2 / --no_fc` 冻结对应层（lr 置 0，该层权重不变）。

### 6.3 多线程数据加载（`--threads`，默认 8）

- 仅作用于 **GPU 路径**的读图 + 预处理 + 离线增强。
- `--threads 1` = legacy 单线程；`--threads 0` = auto（`hardware_concurrency()`）；
  上限被裁剪为样本数。
- 静态分块：把样本下标区间均分给各线程，各写线程本地缓冲，主线程 join 后按顺序无锁合并，
  避免共享写竞争。

### 6.4 One-Cycle 动态学习率（默认关闭）

- `--onecycle` 开启；关闭时 `lrScaleFactor` 恒返回 `1.0f`，训练与固定 LR **逐位一致**。
- 参数：`--lr_warmup N`（0=auto=`max(1,epochs/5)`）、`--lr_min F`（0.01）、`--lr_peak F`（1.0）。
- 曲线：前 `warmupSteps` 从 `base*minFactor` **线性升**到 `base*peakFactor`；
  其后 **余弦退火**回 `base*minFactor`。
- 粒度：**GPU 为 per-batch**（step = epoch·stepsPerEpoch + batchIndex），
  **CPU 为 per-sample**（step = epoch·N + 样本序号）。
- 非法值会被 clamp 并告警：`warmup<0→auto`、`min<0→0`、`peak<=0→1`、`min>peak→min=peak`。

### 6.5 GPU 端 loss/acc 累加（B1 优化）

- `kloss_acc` 在 GPU 上累加本 epoch 的 loss / 正确数 / 各类预测计数。
- **每个 epoch 只做一次 D2H** 取回这些统计量，消除训练循环中每 batch 的
  同步 + host 串行计算。

### 6.6 训练保存时机

- `bkc` 结束保存到 `--model`（默认 `model.txt`）。
- 每个 epoch 满足 `epoch % 10 == 9`（即第 9、19、29… 轮）额外保存 `model_epochN.txt`。
- `bkc` 启动时若 `--model` 已存在则**加载后继续训练**，否则随机初始化。

### 6.7 CPU / GPU 数值差异

`--batch 1` 时 GPU 与 CPU 的结果可能略有不同（阈值/累加顺序），默认 batch=128 时影响很小。

---

## 7. BatchNorm（重点）

### 7.1 定义与数据

- 两个 BN 层，分别位于 Conv1 与 Conv2 之后、ReLU 之前。
- 真 BatchNorm：`y = γ·(x - μ_batch) / sqrt(σ²_batch + ε) + β`，统计量**跨 batch 计算**。
- 每组 4 段：`γ`(可训练)、`β`(可训练)、`running_mean`(统计量)、`running_var`(统计量)。
- **常量**（CPU/GPU/文件三处一致）：`eps = 1e-5`、`BN_MOMENTUM = 0.1`。
- **方差口径**：归一化用**有偏**方差（÷N）；写入 running_var 时用**无偏**（×N/(N-1)）。
- 初始化：`γ=1, β=0, running_mean=0, running_var=1`（初始为恒等映射）。
- `running_mean/running_var` **不是训练参数**：`isTrainableParam` 判定后优化器跳过它们；
  `γ/β` 正常训练，使用对应卷积层的 `lr_conv1 / lr_conv2`。

### 7.2 硬规则（违反会直接报错退出）

| 规则 | 说明 |
| --- | --- |
| BN 只能 **GPU 训练** | `bkc ... --BN` 最终走 CPU（`--cpu` 或无 GPU）报错：`CPU 训练不支持 BN，请用 --gpu` |
| 推理支持 BN | CPU / GPU test、单图、GUI 均可推理 BN 模型（用 running 统计量，不更新） |
| `BN:ON` 模型必须传 `--BN` | 加载 BN 模型未传 `--BN` → 报错 |
| 非 BN 模型不可传 `--BN` | 传了 → 报错 |
| 集成不可混合 | 多个模型中 BN 与非 BN 混用 → 报错 |
| `--BN` 只能在 `gen` 时决定 | 不能在 `bkc`/`test` 阶段给非 BN 模型"启用"BN |

- 训练前向：`kbn_stats`(batch 统计) → `kbn_apply` → `kbn_update_running`。
- 推理前向（eval）：直接用 `running_mean/var` 做 `kbn_apply`，**不更新** running。
- 判断链：`peekModelDims` 解析 `#SIG` 得到 `modelBn`，与命令行 `--BN`（`g_useBN`）
  在 `checkBnCliAgreement` 中比对。

### 7.3 旧模型兼容

- **无 `#SIG`** 或头部**无 BN 字段**的旧模型，一律归一化为 `BN:OFF`。
- `net_spec.h` 中 `bn=false` 时所有 BN 段数量为 0，偏移自然塌缩，
  非 BN 路径布局与改造前**逐位一致**。
- `loadModel` / `peekModelDims` 仅在**实际读到** `BN1`/`BN2` 行时才读取，
  旧模型无此行自然回退。

---

## 8. 模型文件格式

模型文件有**两种格式**，二者承载同一份网络结构与权重、可无损互转：

- **acm 二进制（默认）**：固定 1024 字节头部 + fp32 数据区，体积更小、加载更快；
- **txt 文本**：字段用空格分隔、一行一段，便于查看与手改。

格式由**文件名后缀**与 `--txtmodel` / `CNN2_TXTMODEL` 共同决定，详见 [8.5](#85-acm-二进制格式默认--txtmodel-与后缀规则)。
下文 8.1–8.4 描述的是 **txt 文本格式**的布局。

### 8.1 头部签名 `#SIG`（强制写）

首行**始终**写 `#SIG`（自 P0 起不再依赖 `--P`）。格式：

```
#SIG [gpu=<GPU 名> | host=<计算机名> | time=<ISO 时间> | ]BN:ON|OFF | depth=2
```

- 加 `--P`：写全 `gpu / host / time` 三字段（GPU 名可能含空格，字段间用 ` | ` 分隔）。
- 不加 `--P`：只写 `BN:ON|OFF | depth=2`。

### 8.2 各维度行

| 段 | 内容 |
| --- | --- |
| `K1 o1 i1 h1 w1` | Conv1：`o1=c1, i1=1, h1=w1=k1`，随后 `o1*i1*h1*w1` 个 float（同一行） |
| `B1 n` | n 个 float（同一行） |
| `BN1 c1`（仅 BN） | 随后 4 行：c1 个 γ、c1 个 β、c1 个 mean、c1 个 var |
| `K2 o2 i2 h2 w2` | Conv2：`o2=c2, i2=c1, h2=w2=k2`，随后 `o2*i2*h2*w2` 个 float |
| `B2 n` | n 个 float |
| `BN2 c2`（仅 BN） | 随后 4 行：γ / β / mean / var，各 c2 个 |
| `P<i> stride pad pool`（阶段2，可选） | 每层 `B<i>`/`[BN<i>]` 之后；缺失按 `1 0 2`。`pool∈{0,2}`。depth>2 时每层都写 |
| `FC outDim inDim` | 随后 `outDim` 行、每行 `inDim` 个权重；最后一行 `outDim` 个偏置 |

> 每层 `K<i>` 的输入通道 `i<i>` 必须等于上一层输出通道 `o<i-1>`，否则 `peekModelDims` 拒绝。

### 8.3 完整样例

**非 BN 模型（`--P`，默认 32/64）**：

```
#SIG gpu=NVIDIA Tesla P100-PCIE-16GB | host=WORKSTATION | time=2026-10-03T12:00:00 | BN:OFF | depth=2
K1 32 1 5 5
0.123 -0.045 ... (共 800 个 float)
B1 32
0.0 0.0 ... (共 32 个)
K2 64 32 3 3
0.01 -0.02 ... (共 18432 个)
B2 64
0.0 0.0 ... (共 64 个)
FC 10 2304
... (10 行，每行 2304 个)
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```

**BN 模型（不加 `--P`）**：

```
#SIG BN:ON | depth=2
K1 32 1 5 5
... (800 个)
B1 32
... (32 个)
BN1 32
1.0 ... (32 个 γ)
0.0 ... (32 个 β)
0.0 ... (32 个 running_mean)
1.0 ... (32 个 running_var)
K2 64 32 3 3
... (18432 个)
B2 64
... (64 个)
BN2 64
1.0 ... (64 个 γ)
0.0 ... (64 个 β)
0.0 ... (64 个 running_mean)
1.0 ... (64 个 running_var)
FC 10 2304
... (10 行)
... (10 个偏置)
```

### 8.4 旧模型兼容

`loadModel` / `peekModelDims` 首行检测逻辑相同：

- 若首个非空行以 `#` 开头 → 视为签名行并跳过，同时解析 `BN:` / `depth=` / `ver=`；
- 否则回退到文件开头（兼容旧格式）；
- 顺带剥离 UTF-8 BOM（防御性）。

旧模型（无 `#SIG`）按 `BN:OFF`、`depth=读到的层数`、`ver=1`、`P<i>` 默认 `(1,0,2)` 处理。
`depth=2` 文件即使带 `#SIG` 也保持旧头部格式（不写 `ver=2`）。

### 8.5 acm 二进制格式（默认）、`--txtmodel` 与后缀规则

**默认保存格式为 acm 二进制。** 文件布局为「固定 1024 字节头 + fp32 数据区」：

| 区段 | 偏移 | 内容 |
| --- | --- | --- |
| 标量区 | `[0, 64)` | 魔数 `ACMN`、`headerVersion`、格式版本 `formatVersion`（= `#SIG` 的 `ver`）、**精度字段 `precision`（占位，当前仅 fp32，预留 fp16/int8）**、flags、BN、depth、输入 H/W、类别数、flat、参数量 `total`、数据字节数、签名长度、数据区 CRC32 |
| `#SIG` 原文 | `[64, 320)` | 与 txt 首行**逐字符一致**的 `#SIG` 串（NUL 填充） |
| 层表区 | `[320, 800)` | 24 条 × 20B 的层记录（`outC/inC/kH/kW/stride/pad/pool/bn/res/resPos/resFrom`） |
| 保留区 | `[800, 1024)` | 零填充（预留扩展） |
| 数据区 | `[1024, …)` | `total` 个 float32，顺序与 `flattenParams` 一致 |

- **文件大小恒为 `1024 + total*4` 字节**（如默认 2 层网 `total=42378` → `170536` 字节）；
  acm 与 txt 保存的是**同一份权重与结构**，可无损互转。
- **读取侧恒按魔数自动识别**（`ACMN` → acm，否则按 txt），**不受 `--txtmodel` 影响**；
  后缀为 `.acm` 但魔数不符时判为损坏并报错。
- 头部 `precision != 0`（非 fp32）时，`read` / 加载均报「暂不支持」。

**`--txtmodel`（含环境变量 `CNN2_TXTMODEL`）**：以纯文本格式保存模型。

- `--txtmodel` 强制「保存」为 txt；启动时若环境变量 `CNN2_TXTMODEL` 非空且不为 `0`，
  同样把**默认保存格式**切回 txt（显式 `--txtmodel` 优先级最高）。
- 未给 `--txtmodel`/env 时默认保存为 **acm**。

**文件名后缀规则**（保存格式判定，优先级：后缀 > 开关）：

| 后缀 | 保存格式 |
| --- | --- |
| `.txt` | **txt**（后缀决定，等同隐含 `--txtmodel`） |
| `.acm` | **acm**；若同时给了 `--txtmodel`/env → **报错退出**（`.acm` 表示二进制，与文本冲突） |
| 无 / 其它后缀 | 看开关：默认 acm，`--txtmodel`/env 时 txt |

**默认模型名**：未显式给 `--model` 时，默认名随格式——默认 `model.acm`，`--txtmodel` 时为 `model.txt`。
同理，程序自行生成的路径后缀也随格式：每 10 轮的 `model_epochN`、`--epsave` 的 `ep_<epoch>`、
`--val` 的 `<stem>.best` 都会写成 `.acm` 或 `.txt`（如默认 `ep_0.acm`、`model.best.acm`）。

**`read` 命令**：轻量查看模型文件（不训练、不推理），自动识别 acm/txt：

```bat
CNN2.exe read model.acm
CNN2.exe read model.txt
CNN2.exe read --model model.acm
```

acm 输出示例（默认 2 层网）：

```
模型文件: model.acm
格式: acm(二进制)  头版本=1  ver=1  精度=fp32  flags=0x3  文件大小=170536 字节
#SIG 等价: BN=OFF depth=2 ver=1
网络: 输入 32x32  类别 10  深度 2  参数量 total=42378
逐层:
  L1: K 32x1x5x5  B 32  P(1,0,2)
  L2: K 64x32x3x3  B 64  P(1,0,2)
FC: 10x2304
```

成功退出 0，失败（文件不存在 / 损坏 / 精度不支持）退出 1。

---

## 9. 命令行参数总表

> 解析规则：`--key value`（下一个 token 不以 `--` 开头时作为值）；否则该 `--flag` 值为 `"1"`。
> 因此单独写 `--prep` 等价于 `--prep 1`，会**关闭全部预处理步骤**（注意）。

### 9.1 模式与位置参数

| 参数 | 说明 |
| --- | --- |
| `<路径> gen` | 生成随机初始化模型并保存到 `--model` |
| `<训练目录> bkc` | 训练 |
| `<测试目录> test` | 测试 / 评估 |
| `<单图.txt>`（无模式） | 单图推理 |
| `read <模型文件>` | 读取/查看模型文件（自动识别 acm/txt，见 8.5） |
| `--gui` | GUI 模式（可出现在任意位置，优先于其它模式） |
| `tui` / `--tui` | 终端 TUI 指令台（与 `--gui` 并列的交互模式，opt-in）；在终端全屏界面中输入 CNN2 指令并回车执行，内建 `help`/`clear`/`quit`/`cd`/`history`/`status` |

### 9.2 通用 / 模型

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model <path>` | `model.acm` | 模型路径（默认随格式：acm 时 `model.acm`，`--txtmodel` 时 `model.txt`）；`test` 时可用逗号分隔多模型做集成 |
| `--txtmodel` | off | 以纯文本格式保存模型（含环境变量 `CNN2_TXTMODEL`；后缀判定见 8.5） |
| `--fpng <path>` | 无 | GUI 启动时加载到画布的 32×32 txt |
| `--c1 N` | 32 | Conv1 输出通道（`gen`/`bkc` 生效；加载时以文件为准） |
| `--c2 N` | 64 | Conv2 输出通道（`gen`/`bkc` 生效；加载时以文件为准） |
| `--depth N` | 2 | 网络深度（`gen`/`bkc` 新建生效）；depth>2 用 3.6 默认 schedule |
| `--BN` / `--bn` | off | 启用 BatchNorm（语义见第 7 节） |
| `--P` | off | 保存模型时签名写完整环境字段（见第 11 节） |

### 9.3 训练超参

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--epochs N` | 10 | 训练轮数 |
| `--batch N` | 128 | batch 大小（GPU 训练 / GPU 推理用） |
| `--lr_fc X` | 0.0001 | 全连接学习率 |
| `--lr_conv1 X` | 0.0001 | Conv1 学习率 |
| `--lr_conv2 X` | 0.0005 | Conv2 学习率 |
| `--momentum B` | 0.9 | SGD momentum 系数，0=关闭退化为朴素 SGD |
| `--nesterov` | off | 启用 Nesterov look-ahead |
| `--no_conv1` | — | 冻结 Conv1 |
| `--no_conv2` | — | 冻结 Conv2 |
| `--no_fc` | — | 冻结 FC |
| `--gpu` | 自动 | 强制 GPU |
| `--cpu` | 自动 | 强制 CPU（与 `--gpu` 同时给出时 CPU 优先） |
| `--seed N` | `time(NULL)` | 全局随机种子：初始化 / 采样 / 每 epoch 洗牌 / 增强（见 15.6 的可复现保证） |
| `--threads N` | 8 | GPU 数据加载线程数；0=auto；1=单线程 legacy |
| `--time` | off | 输出每 epoch 与总耗时 |
| `--log <file>` | off | 追加训练日志：每 epoch 一行 CSV + 每次存档一行 CHECKPOINT（见下方说明） |
| `--val <目录>` | off | 独立验证集：每 epoch 末评估并打印 `val acc`；配合 `--log` 追加 `val_acc` 列；并保存最优模型到 `<model>.best`（见下方说明） |
| `--checkdata` | off | 数据校验模式：`cnn2 <目录> --checkdata` 扫描并报告文件数/每类计数/非法文件后退出 |
| `--strict-data` | off | 校验发现非法文件时报错并非零退出（配合 `--checkdata`，或用于 `bkc` 启动前校验训练/验证目录） |
| `--wd F` | 0.0 | L2 weight decay（coupled，`g += wd*param`）；仅作用于**乘法权重**（卷积核 / 投影权 / FC 权），BN `γ/β` 与 bias 不衰减；默认 0，逐位不变（见下方说明） |
| `--epsave [目录]` | off | 每轮保存权重到 `ep_<epoch>.txt`；目录省略时默认 `EPTSET\<模型名>\`（相对当前工作目录）；仅 `bkc` 生效（见下方说明） |
| `--epsn N` | 1 | 每轮存档间隔，默认 1（每轮）；`--epsn 5` 即每 5 轮存一次；仅 `--epsave` 开启时生效 |

**L2 weight decay（`--wd`，阶段7）**：coupled L2，即在梯度上叠加 `g += wd*param`：

- 仅作用于**乘法权重**：卷积核 `K<i>`、1×1 投影权 `PW<i>`、全连接权 `FCW`；
  **BN 的 `γ/β` 与各段 `bias` 不衰减**（与「权重衰减只作用于权重」的常规语义一致）。
- `--wd` 默认 `0.0`，此时不进入叠加路径，训练与输出**逐位不变**（位保护）；
  `--wd 0` 与不传 `--wd` 结果逐字节相同（由回归 `wd_bp` 用例守护）。
- 生效后若设置过大可能导致震荡甚至发散，回归 `wd_active` 用 `--wd 0.01` 断言 exit=0 且 loss 有限。
- **调参建议**：4 层式小网络建议从 `1e-4 ~ 1e-3` 起步，配合 `--val` 观察验证集泛化，
  避免只盯训练 loss 而过拟合。

**每轮存档（`--epsave` / `--epsn`，阶段7）**：`--epsave` 开启后，`bkc` 在每个满足
`epoch % epsn == 0` 的 epoch 结束时把当前权重保存为 `<目录>\ep_<epoch>.txt`（`epoch` 从 0 计）：

- 目录省略（写成裸 `--epsave`）时默认 `EPTSET\<模型名>\`（`<模型名>` 为 `--model` 的 basename 去扩展名），
  相对**当前工作目录**创建；也可显式给出绝对 / 相对目录。
- `--epsn N` 为保存间隔，默认 1（每轮）；如 `--epsn 5` 则每 5 轮存档一次，仅 `--epsave` 开启时生效。
  非法值（`< 1`）会被 clamp 为 1 并在 stderr 给出警告。
- 该存档与既有的「每 10 轮保存 `model_epochN.txt`」并存、互不影响；**未传 `--epsave` 时不创建目录、不改变任何行为**。

**训练日志（`--log`，阶段4.3 可观测性）**：开启后向 `<file>` 以**追加模式**写入（每条 `flush`，
崩溃也能留痕），用于回溯 / 画曲线 / 审计。内容为纯 ASCII CSV（无 BOM）：

- **表头**（仅当文件为空时写首行）：
  `epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec`
- **每 epoch 一行指标**：`loss`/`acc` 为本 epoch 平均 loss 与准确率（与 stdout 同源）；
  `lr_conv1/lr_conv2/lr_fc` 为该 epoch **末实际生效**学习率（非 one-cycle 时即基础 lr，
  one-cycle 时为退火后值）；`epoch_sec` 为该 epoch 墙上耗时（秒）。
- **每次保存模型一行元数据**：
  `CHECKPOINT <epoch> <path> <size_bytes> <time>`，其中 `time` 为本地
  `YYYY-MM-DDTHH:MM:SS`。用「文件大小 + 保存时刻」而非重算 sha256（成本低），
  使「模型文件 ↔ 指标 ↔ 时间」可追溯。`epoch=-1` 表示 `epochs<=0` 的边界情形。

CPU 与 GPU 两条训练路径都会写日志；**未传 `--log` 时不创建文件、不改变任何输出**。

**独立验证集（`--val <目录>`，阶段4.4）**：用于**消除「用 test 集调参」的评估污染**。
开启后，`bkc` 每 epoch 结束后在验证集上评估一次准确率：

- **预处理/推理与 `test` 完全同源**（`loadImagePreprocessed(g_prep)` + `forwardConv` + `fc.forward` + softmax），
  **不做任何增强**；验证集只读，不参与反向/更新，不改变训练状态。
- 输出：每 epoch 打印 `val acc <值> (<correct>/<total>)`（stdout）。
- 日志：配合 `--log` 时，表头追加一列 `val_acc`（即 8 列），每 epoch 行同步写入；未传 `--val` 时
  日志仍为原 7 列，逐字节不变。
- **最优模型**：每当 `val acc` 严格变好，就把当前权重保存到 `<model>.best`（并写一行 CHECKPOINT）。
  默认不做早停，训练仍按 `--epochs` 跑满，避免改变既有训练语义。
- `--val` 与训练目录相同（`val=train`）时会打印警告，提示指标仅作占位、不代表泛化。
- **未传 `--val` 时行为完全不变**（训练、stdout、日志、存档均与改动前一致）。

**数据校验（`--checkdata` / `--strict-data`，阶段4.4）**：`cnn2 <目录> --checkdata` 扫描目录并报告
**文件数、每类计数、非法文件清单**。非法判定：

- 文件名首字符非数字；
- 空文件；
- 行数 ≠ 32（非 32×32）或存在非 32 列的行；
- 内容含非 `0`/`1` 字符。

不带 `--strict-data` 时仅报告并退出 0；带 `--strict-data` 时若发现非法文件则报错并退出 1。
`--strict-data` 也可用于 `bkc`：训练前对训练目录与 `--val` 目录做校验，发现非法即中止。

### 9.4 One-Cycle 学习率

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--onecycle` | off | 启用 One-Cycle（默认关闭时固定 LR） |
| `--lr_warmup N` | 0=auto | warmup 轮数，auto=`max(1,epochs/5)` |
| `--lr_min F` | 0.01 | 最小 lr 因子（`base*F`） |
| `--lr_peak F` | 1.0 | 峰值 lr 因子（`base*F`） |

### 9.5 数据混合与增强

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data2 <目录>` | 无 | 第二套数据目录，与主目录混合训练 |
| `--mix <mode>` | `balanced` | 混合策略：`balanced`(域均衡分层) / `full`(全量投喂) |
| `--full` | — | 等价 `--mix full` |
| `--aug` | off | 数据增强总开关 |
| `--aug_k K` | 2 | 离线扩增副本数 |
| `--aug_shift X` | 3 | 随机平移 ±X px |
| `--aug_rotate X` | 12 | 随机旋转 ±X 度 |
| `--aug_scale X` | 0.1 | 缩放抖动半幅 |
| `--aug_elastic A` | 0 | 弹性形变 α（0=关） |
| `--aug_elastic_sigma S` | 4.5 | 弹性位移场 σ |
| `--aug_thick P` | 50 | 笔画粗细触发概率 %（0=关） |

混合训练细节：无 `--data2` 时直接用主目录全部样本；
有 `--data2` 且 `balanced` 时按 `(域, 标签)` 分层，取各非空层最小数量为目标下采样后合并打乱；
`full` 时两域全部样本合并打乱，不下采样。

### 9.6 测试 / 集成 / TTA / 判错

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--gpu` | off | test 时启用批量 GPU 推理（test 默认 CPU） |
| `--outerror` | off | 输出判错样本清单（屏幕 + `outerror.txt`） |
| `--fuse <mode>` | `soft` | 融合：`soft`/`logit`/`geom`/`vote`/`gate`/`wbest` |
| `--weights w1,w2,...` | 全 1 | 每模型权重（逗号分隔） |
| `--fuse_margin M` | 0.0 | `gate` 模式独醒者反选阈值 |
| `--fuse_temp T` | 1.0 | 概率温度缩放（`p^(1/T)` 后逐模型归一化） |
| `--tta` | off | 测试时增强（仅 test，log 概率空间平均） |
| `--tta_k K` | 5 | TTA 变换数量（最多 11） |

### 9.7 预处理 / 诊断

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--prep crop,deskew,resize,center,thick` | 见下 | 显式指定时**先全部关闭**再按清单开启；`none` 或 `1` 关闭全部 |
| `--prep_target N` | **28** | resize 目标最长边，clamp 到 16–30 |
| `--prep_stretch` | off | 拉伸到 target×target |
| `--prep_ink X` | 0.20 | thick 目标墨迹比，0=关闭 |
| `--prep_deskew_max D` | 15 | deskew 角度限幅（度） |
| `--diag` | off | 打印预处理前后统计 |

不加 `--prep` 时的默认步骤为 **crop + resize + center**（deskew、thick 关闭）。

### 9.8 CLI 归一化与子命令（本版新增）

- **子命令式**：新增 `cnn2 train <目录>` / `cnn2 test <目录>` / `cnn2 gen <路径>` /
  `cnn2 infer <图>` / `cnn2 gui`，等价于旧式 `cnn2 <路径> <模式>`。判别规则：仅当
  `argv[1]` 恰为子命令词**且 `argv[2]` 不是模式词（`gen`/`bkc`/`test`）**时按子命令解析，
  否则沿用旧式。故 `cnn2 train bkc` 仍解析为**路径 `train`、模式 `bkc`**（不是子命令）。
- **键名短横线写法**：`--` 后键名大小写不敏感、`-` 与 `_` 等价，如 `--lr-fc` ≡ `--lr_fc`、
  `--strict_data` ≡ `--strict-data`；`--bn` ≡ `--BN`。
- **未知参数警告**：未知 `--key` 打印警告并忽略（含 did-you-mean 建议），不退出。
- **`--out <路径>`**：仅 `gen` 模式生效，等价 `--model`（未显式给 `--model` 时才应用）。

---

## 10. 测试 / 评估（集成 / 融合 / TTA / 判错）

### 10.1 单模型 / 多模型集成

- `--model a.txt,b.txt,c.txt` 逗号分隔多个模型。
- 集成要求所有模型**维度一致且 BN 状态一致**，否则报错退出（异构集成暂不支持）。
- 每模型对每样本输出 softmax 概率，存入 `allProb[(m*N+i)*10+c]`，再按 `--fuse` 逐样本定预测。
- `M=1` 时行为与旧单模型完全一致。

### 10.2 融合模式

| 模式 | 语义 |
| --- | --- |
| `soft`（默认） | 各模型概率加权平均后取 argmax |
| `logit` | 加权 log 概率平均后取 argmax |
| `geom` | **当前实现与 `logit` 走同一 log 累加路径**（以代码为准） |
| `vote` | 每模型一票，多数票；平票取权重最高模型 |
| `gate` | 3 模型场景门控：三票一致取该票；2:1 时若独醒者平均概率 > 多数者平均概率 + `margin` 则反选；否则取多数；三票分散取权重最高模型。`M≠3` 时退化为加权 soft |
| `wbest` | 直接取权重最大模型的 argmax |

`--fuse_temp T`：先对概率做 `p^(1/T)` 并按模型重归一化，再进入上述融合。

### 10.3 TTA（测试时增强）

- 仅 `test` 生效，对**已预处理**的 32×32 二值图做轻微几何变换，
  **绝不再调用 `preprocessImage`**（否则会抵消变换）。
- 固定变换序列（由温和到更强，t0 恒为原图），共 11 个：

  ```
  t0 原图
  t1 右移1px  t2 左移1px
  t3 +8°      t4 -8°
  t5 右移2px  t6 左移2px
  t7 +12°     t8 -12°
  t9 下移1px  t10 上移1px
  ```

- `--tta_k K` 取前 K 个，clamp 到 [1,11]，默认 5。
- 平均在 **log 概率空间**：对每个变换求 `log p`，累加后 softmax 回概率，再进入融合。
- GPU 路径下 t0 原图零拷贝；其它变换按需重新生成数据。

### 10.4 `--outerror` 判错清单

- 屏幕打印并以 `outerror.txt`（当前工作目录）落盘。
- 每条含：文件路径、真值、预测、置信度（融合后预测类加权平均概率）、真值概率。
- 末尾输出判错总数；无判错时打印"（未发现判错样本）"。

---

## 11. 模型签名 #SIG 与 --P

### 11.1 字段

```
#SIG [gpu=<name> | host=<name> | time=<ISO> | ]BN:ON|OFF | depth=N [| ver=2]
```

- `--P`：写全 `gpu / host / time`。
- 不加 `--P`：仅 `BN` 与 `depth`。
- `depth` 为实际层数（阶段2 起支持 `[1, MAX_DEPTH]`）；`depth>2` 或存在非默认 `P<i>`
  时追加 ` | ver=2`，否则维持旧格式 `depth=2`（无 `ver`）。
- 加载时以**实际读到的层数**为准（`#SIG` 的 `depth` 仅作参考）。

### 11.2 解析规则（`parseSigFields`）

- 按 ` | ` 切分、trim，大小写不敏感，兼容 `BN:`/`BN=`、`depth:`/`depth=`、`ver:`/`ver=`。
- `BN` 非 `on/off` → 按 `BN:OFF` + 警告；`depth` 非法 → 2 + 警告。
- `ver` 字段**仅作诊断**：仍解析并做合法性检查，非法时打印原警告（按 `ver=1` 处理），
  但不再存回 `NetSpec`（`NetSpec::ver` 字段已随重构移除）。
- 缺省（含旧签名无 `BN:` 字段）→ `BN:false, depth=2`。

### 11.3 `normalizeSignature`（续训行为）

续训（`bkc` 加载已有模型）保存时**不原样复用旧签名**：

- **保留**旧签名中的设备/时间等字段（若有），**强制刷新** `BN:` 与 `depth=`/`ver=`；
- 旧签名若没有设备字段，新签名就生成「仅 BN+depth[+ver]」格式；
- 这样既能延续设备信息，又能保证 BN/depth/ver 与实际模型一致。

---

## 12. 构建

### 12.1 环境变量与工程

- `CNN2.vcxproj` 通过 `CUDA 12.9.props/targets` 引入 CUDA 集成，
  依赖环境变量 `CUDA_PATH_V12_9`（指向 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`）。
- PlatformToolset v143，C++17，x64 Release。
- CUDA 代码生成：`compute_60,sm_60; compute_61,sm_61; compute_60,compute_60`。
- 会看到 `nvcc warning: ... prior to '<...>_75' ...`，属正常提示。
- TUI 依赖 FTXUI：源码已 vendored 到 `third_party/ftxui/`（含鼠标补丁），**无需额外安装**，
  构建命令不变。

### 12.2 MSBuild 命令

```bat
set "CUDA_PATH_V12_9=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
"D:\vc++\MSBuild\Current\Bin\MSBuild.exe" CNN2.vcxproj /p:Configuration=Release /p:Platform=x64
```

> MSBuild 实际路径以本机 VS 安装为准（本仓库实测使用 .NET Framework MSBuild 17.14）。
> 输出：`x64\Release\CNN2.exe`。构建完成后按第 1.3 节把 exe + 4 个 dll 拷入 `deploy\`。

### 12.3 发布 / 可复现构建（`release.ps1`）

项目根目录的 `release.ps1` 提供「一键发布」：**读版本 → 编译 → 打包 → 压缩**。
固定工具链为 **Visual Studio 2022 / MSVC v143、CUDA 12.9、Release x64**。

```powershell
.\release.ps1                       # 完整发布：编译 + 打包 + 压缩
.\release.ps1 -SkipBuild            # 仅用现有 x64\Release\CNN2.exe 重新打包（不编译）
.\release.ps1 -NoZip                # 只生成目录，不压缩 zip
.\release.ps1 -BuildTimeoutSec 1800 # 自定义编译超时（秒，默认 1800 = 30 分钟）
```

脚本依次执行：

1. 从 `version.h` 解析程序版本 `CNN2_VERSION`（并读出模型格式版本 `CNN2_MODEL_VER`）；
2. 设 `$env:CUDA_PATH_V12_9` 后调用 `D:\vc++\MSBuild\Current\Bin\MSBuild.exe`
   （缺失时回退 PATH 中的 `MSBuild.exe`）编译 Release x64；编译带**超时保护**，
   超过 `-BuildTimeoutSec` 即杀掉整棵进程树并按失败退出，避免工具链挂起；
3. 生成 `dist\CNN2-<版本>-win-x64\`，内含：
   - `CNN2.exe`；
   - 4 个运行时 dll：`cudart64_12.dll`、`msvcp140.dll`、`vcruntime140.dll`、
     `vcruntime140_1.dll`（优先取自 `deploy\`；缺失时回退 `x64\Release\`，
     两处都没有则**明确报错并列出缺少的 dll**，不会静默漏拷）；
   - `README.md`、`CHANGELOG.md`、`BUILDINFO.txt`；
4. 用 `Compress-Archive` 压缩为 `dist\CNN2-<版本>-win-x64.zip`，并打印产物路径与 SHA256。

#### `BUILDINFO.txt` 字段

| 字段 | 含义 |
| --- | --- |
| `Version` | 程序版本（来自 `version.h` 的 `CNN2_VERSION`） |
| `ModelFormatVer` | 模型格式版本（`ver=N`，来自 `CNN2_MODEL_VER`） |
| `BuildDate` | 构建时间（取自 `CNN2.exe` 的 `LastWriteTime`，ISO 8601） |
| `BuildHost` / `BuildUser` | 构建主机名 / 用户名 |
| `GitCommit` | git 提交号；非 git 仓库时为占位 `n/a (非 git 仓库)` |
| `Toolchain` | 编译器工具集（MSVC v143 / VS2022 / C++17） |
| `MSBuildVersion` | MSBuild 产品版本 |
| `CUDA` | CUDA 版本 |
| `Configuration` | 固定 `Release x64` |
| `Executable` / `ExeSizeBytes` | 可执行文件名 / 字节数 |
| `ExeSHA256` | `CNN2.exe` 的 SHA256（大写十六进制） |
| `PackageDate` | 打包时间 |

#### 如何校验产物 SHA256

```powershell
Get-FileHash .\dist\CNN2-1.0.0-win-x64\CNN2.exe -Algorithm SHA256
Get-FileHash .\dist\CNN2-1.0.0-win-x64.zip       -Algorithm SHA256
```

比对结果应与脚本结尾打印、以及 `dist\CNN2-<版本>-win-x64\BUILDINFO.txt` 中的
`ExeSHA256` 一致。发布/分发时以该 SHA256 作为产物指纹。

> **关于「可复现」的边界**：脚本固定了**工具链、工程配置与打包步骤**（VS2022 v143 +
> CUDA 12.9 + Release x64），并在 `BUILDINFO.txt` 固化 `ExeSHA256` 供核对。
> 但 MSVC 默认会在 PE 头写入时间戳，**增量未重链接时 SHA256 不变；强制重新链接时
> SHA256 可能变化**，属正常现象。需要逐位可复现时，应在工程中启用 `/Brepro`
> （本阶段未改动 `CNN2.vcxproj`）。

> `release.ps1` 只做编译与打包，**不改任何核心源码，也不改 `build_and_test.ps1`**；
> 发布前的回归测试仍请运行 `build_and_test.ps1` 并确保全绿。

---

## 13. 工程文件说明

| 文件 | 职责 |
| --- | --- |
| `main.cpp` | 主程序：参数解析、模式分发、模型 save/load、预处理入口、TTA、集成融合、CPU 训练、test 评估、GUI 分流 |
| `file.h` | 路径 / 目录 / 格式探测工具 + acm 二进制模型头部编解码（含 CRC32）（header-only） |
| `net_spec.h` | `NetSpec` 结构 + `computeNetSpec` / `makeDefaultNetSpec`（及 `makeResNetSpec`/`makeProjResNetSpec`/`makePreactResNetSpec`）/ `validateNetSpec` / `isTrainableParam`（host/device 共用） |
| `conv.h` | `ConvMap` 与 CPU 前向 `conv` / `relu` / `Pool2x2`；含旧版 16 个 3×3 手工核（已弃用）。**非 inline 全局定义，只能被 main.cpp 包含** |
| `sf.h` | `flatten`、`FullyConnected`（W/b/forward/save/load）、`softmax`、`FC_backward`、`relu_backward`、`pool2x2_backward`、`conv_backward`。**非 inline，只能被 main.cpp 包含** |
| `preprocessing.h` | `PrepConfig`/`PrepStats` 与预处理全部算子、`preprocessImage`、`prepConfigFromArgs`（header-only） |
| `augment.h` | `AugConfig` 与几何/弹性/粗细增强、`augmentRandom`、`augmentOffline`（header-only） |
| `gpu_kernels.cuh` | CUDA 头：`GpuBuffers`、默认规模宏（仅记录用途）、BN 常量、全部 kernel 声明。**只被 .cu 包含** |
| `gpu_kernels.cu` | CUDA kernel 实现（前向 / 反向 / BN / loss 累加 / 参数更新） |
| `gpu_trainer.cuh` | 纯 C++ 对外接口：`LrScheduleConfig`/`lrScaleFactor`、`GpuTrainConfig`、`GpuEpochCallback`、`gpuAvailable`/`gpuName`/`gpuTrain`/`gpuPredict` |
| `gpu_trainer.cu` | GPU 显存管理、数据上传、训练主循环、`gpuTrain` / `gpuPredict` 实现 |
| `GUI.h` / `GUI.cpp` | Win32 手写识别界面；**刻意不包含 conv.h/sf.h/preprocessing.h**（避免 LNK2005） |
| `TuiLauncher.h` / `TuiLauncher.cpp` | TUI 启动入口（`runTuiMode`，纯 C 接口，供 main.cpp 调用）；负责初始化终端并交出页面控制权 |
| `tui/*.h` / `tui/*.cpp` | 终端 TUI 模块：页面布局与内建指令分发（`TuiPage`）、主题（`TuiTheme`）、控制台挂载（`TuiConsole`）、消息模型（`TuiMessage`）、标题栏（`TuiTitleBar`）、聊天视图（`TuiChatView`）、输入栏（`TuiInputBar`）、指令补全（`TuiCommandCompleter`）、子进程执行（`TuiChildProcess`）、FTXUI 适配（`TuiFtxui.h`）与帮助文案（`TuiHelpText.h`） |
| `third_party/ftxui/` | vendored FTXUI 7.0.3（`ftxui.hpp` / `ftxui.cpp`，含鼠标补丁） |
| `CNN2.sln` / `CNN2.vcxproj` | VS 工程 |
| `deploy\*.bat` | 部署/训练/测试批处理脚本（GBK 编码） |

> `CONV.cpp` 为历史遗留文件，不参与当前构建。

---

## 14. 部署脚本

`deploy\` 下脚本均以 `chcp 936`（GBK）运行，双击即可；默认数据目录：

```
训练目录：%USERPROFILE%\trainingDigits（默认取 deploy\..\data\trainingDigits，可通过环境变量 TRAIN 覆盖）
测试目录：%USERPROFILE%\testDigits（默认取 deploy\..\data\testDigits，可通过环境变量 TEST 覆盖）
```

改路径请编辑脚本顶部的 `TRAIN` / `TEST`。

| 脚本 | 用途 |
| --- | --- |
| `1.bat` | 六模型训练（M1–M6）：两阶段 A/B，`target=28`，One-Cycle，`batch 256`，输出 `models\M1..M6.txt`。已存在则跳过 |
| `2.bat` | 在上一步模型基础上**补训练**（P1_ft…P6_ft），`batch 256` +60 epoch |
| `run_train_all.bat` | 一键训练 P1–P6（不同增强配方），输出 `P1_geom_s1001.txt` 等 |
| `prep_test.bat` | 固定配方对比 5 种预处理（默认 / +deskew / +thick / target=28 / stretch），输出 `PREP_*.txt`，并在测试集上打印各配置准确率 |
| `test99.bat` | 快速冲刺 99+：阶段 A 30 轮（batch 128）+ 阶段 B 10 轮低 LR，最后 `test --gpu --outerror` |
| `test99b.bat` | 在 test99 基础上学习率 ×10 |
| `test99c.bat` | 学习率 ×20，阶段 A 40 轮 + 阶段 B 12 轮 |
| `test99d.bat` | One-Cycle 动态学习率测试（warmup 5 + cosine 退火） |
| `test99e.bat` | One-Cycle 阶段 A + 低 LR 阶段 B（C+D 混合） |
| `test99f.bat` | M_best 配方复现（阶段 1：30 轮 lr=0.01；阶段 2：10 轮 lr=0.001） |
| `test99g.bat` | `target=28` + CD 组合配方 |

> `deploy\README.txt` 为旧版用户手册（GBK），仅供部署参考，**参数与行为以本 README 和代码为准**。

---

## 15. 编码规范与陷阱

### 15.1 文件编码

- **含中文的源码文件必须保存为 UTF-8 with BOM**，否则 MSVC 在默认 936 代码页下会误解析
  （尤其 nvcc 编译单元）。
  当前为 UTF-8 BOM：`main.cpp`、`GUI.cpp`、`GUI.h`、`net_spec.h`、`preprocessing.h`、
  `augment.h`、`gpu_kernels.cuh`、`gpu_trainer.cuh`、`gpu_kernels.cu`、`gpu_trainer.cu`；
  TUI 相关：`TuiLauncher.h` / `TuiLauncher.cpp` 与 `tui/*.h` / `tui/*.cpp`。
- `conv.h`、`sf.h` 为**旧版 GBK/ANSI（无 BOM）**，编译可通过，但用 UTF-8 打开时中文会乱码；
  新增/修改这两个文件时请统一为 UTF-8 BOM。
- `deploy\*.bat`、`deploy\README.txt` 为 GBK（`chcp 936`），请勿用 UTF-8 覆盖保存。
- 本 `README.md` 为 UTF-8。

### 15.2 训练 / 测试预处理必须一致

见 4.4。测试时必须带上与训练相同的 `--prep*` 参数，否则精度会下降。

### 15.3 参数偏移同源

`flattenParams` / `scatterParams` / 所有 GPU kernel 一律读 `NetSpec`，
**禁止硬编码偏移**。改动布局必须同步改 `net_spec.h` 的 `computeNetSpec`（及 `makeDefaultNetSpec` 等构造器）。

### 15.4 头文件重复定义

`conv.h` / `sf.h` / `preprocessing.h` / `augment.h` 中存在**非 inline 全局函数/变量定义**：

- 只允许 `main.cpp` 一个 C++ 编译单元包含它们；
- `GUI.cpp` 刻意不包含，避免 LNK2005；
- GPU 侧改用扁平参数接口（`gpu_trainer.cuh`），不直接包含这些头。

### 15.5 Windows 宏陷阱

`windows.h` 会定义 `min`/`max` 宏。比较时用 `(std::min)` / `(std::max)`（带括号防止宏展开）
或三元表达式；`conv.h`/`sf.h` 等已包含 `<algorithm>` 时仍需注意。

### 15.6 `--seed` 的可复现保证

指定 `--seed N` 时，以下随机源全部确定化，**同一 seed + 同一数据 + 同一参数**下
两次运行的模型文件（`gen`/`bkc`）逐字节相同、训练日志（loss/acc）逐位相同：

- **卷积核初始化**（`buildKernels`）：由 `mixSeed(seed, 0x1)` 派生；
- **FC 随机初始化**（`gen` 及 `bkc` 新建模型时的 `fc.randomInit`）：由 `mixSeed(seed, 0x2)` 派生；
- **每 epoch 样本乱序**（CPU 与 GPU）：各自用 `mt19937(mixSeed(seed, 0x5EED ^ epoch))`
  重新播种后 `std::shuffle`，替换原先的 `std::random_device`；
- **数据混合采样**（`buildTrainSet`）与**增强 RNG**：由 `seed` 派生；
- GPU/CPU 并行加载的增强按 `(seedBase, 样本下标)` 派生 RNG，**与线程划分无关**，
  固定 seed 时增强结果可复现。

未指定 `--seed` 时（默认 `time(NULL)`）保持原有的“默认随机”行为。
可复现性由 `build_and_test.ps1` 的 `repro_gen_same` / `repro_gen_diff` / `repro_bkc`
三个用例守护（比对文件 SHA256）。

> 注：`mixSeed(seed, salt)` 定义在 `gpu_trainer.cuh`，CPU 与 GPU 共用，保证同 seed 下
> 两侧派生一致。

### 15.7 `#SIG` 跳过逻辑

`loadModel` / `peekModelDims` 都以「首行是否以 `#` 开头」决定是否跳过签名行；
两者逻辑必须保持一致，修改其一时同步另一处。

### 15.8 其它

- `--cpu` 与 `--gpu` 同时出现时，test 以 `--cpu` 为准，bkc 以 `--cpu` 为准。
- `--batch` 大于样本数时按样本数处理；`<1` 时按 1 处理。
- 模型 `bkc` 续训会覆盖 `--model`，并每 10 轮生成 `model_epochN.txt`。
- 旧模型（无 `#SIG`）按 `BN:OFF` 兼容；给旧模型传 `--BN` 会报错。

---

## 16. 附录：常量速查与常见错误

### 16.1 常量速查

| 常量 | 值 |
| --- | --- |
| 输入 | 32×32，字符 0/1 |
| 默认通道 / 核 | c1=32, k1=5, c2=64, k2=3 |
| 默认 flat / classes | 2304 / 10 |
| 默认参数总量 | 42378（非 BN）/ 42762（BN） |
| 默认 LR | fc=1e-4, conv1=1e-4, conv2=5e-4 |
| 默认 momentum / batch / threads | 0.9 / 128 / 8 |
| One-Cycle | warmup=auto(epochs/5), min=0.01, peak=1.0 |
| BN | eps=1e-5, momentum=0.1，有偏方差归一化，无偏运行方差 |
| 预处理 | target=28, ink=0.20, deskew_max=15 |
| 增强 | k=2, shift=3, rotate=12, scale=0.1, elastic=0, sigma=4.5, thick=50 |
| TTA | k=5，最多 11 个固定变换 |

### 16.2 常见错误信息

| 报错 | 原因 / 处理 |
| --- | --- |
| `CPU 训练不支持 BN，请用 --gpu` | `bkc --BN` 走了 CPU；改用 `--gpu` |
| `模型为 BN:ON，但未传 --BN` | 加载 BN 模型必须显式 `--BN` |
| `传了 --BN，但模型为 BN:OFF` | `--BN` 只能在 `gen` 时决定，不能事后给非 BN 模型启用 |
| `集成: 模型 BN/非BN 混合不被支持` | 集成中不允许混用 BN 与非 BN 模型 |
| `集成: 模型维度不一致` | 集成模型的 c1/k1/c2/k2/flat/classes 必须一致 |
| `缺少 K1 / B1 / K2 / B2 / FC` | 权重文件损坏或格式不符 |
| `无法获取标签` | 文件名首字符不是 0–9 |
| `第 N 行长度不足` / `非法字符` | 图片 txt 不符合 32×32、仅 0/1 的格式 |
| `no valid training data` | 训练目录无有效样本 |
| `--gpu requested but no CUDA device available` | 无可用 CUDA 设备 |

---

## 17. 残差连接（阶段3：identity skip）

### 17.1 语义

- 层可选带一个 identity 残差，指向**更早的同形状层**：
  `y = pool(relu(conv(x))) + layer[resFrom].out`
- 要求源层与本层最终输出形状严格一致（同 `outC` / 池化后 `H` / 池化后 `W`），
  否则 `validateNetSpec` 干净报错。
- **本阶段只实现 identity；1×1 投影残差（res=2）暂不支持，遇到直接报错。**
  理由：保持非残差路径逐位不变、不引入额外投影权重段，风险最低，
  且已足以表达 ResNet/WRN basic block（block 内 pool=0、同通道、同空间）。

### 17.2 文件格式

| 字段 | 说明 |
| --- | --- |
| `R<i> res resFrom`（阶段3，可选） | 仅残差层写。`res=1` 为 identity；`resFrom` 为 0-based 源层号（`0 <= resFrom < i`）。旧文件无此行即无残差 |
| `#SIG ... | ver=2` | 残差属 v2：存在残差时头部追加 `ver=2`；未用残差照旧不写，逐字节兼容 |

示例（depth=3 残差网）：
```
#SIG BN:OFF | depth=3 | ver=2
K1 32 1 3 3
B1 32
P1 1 1 2
K2 32 32 3 3
B2 32
P2 1 1 0
K3 32 32 3 3
B3 32
P3 1 1 0
R3 1 0
FC 10 8192
```

### 17.3 生成与 CLI

- `gen --res --depth N` / `bkc ... --res`：生成残差网络。
  全层同通道 `--c1`；layer0 为 stem（pool=2），其余 pool=0，末层残差指向 layer0。
- **默认（不加 `--res`）不生成残差**，depth=2/3 结构与数值 golden 保持不变。
- 残差网络的 CPU/GPU 前向与反向（梯度分流）均支持；CPU 训练仍不支持 BN。

---

## 18. ResNet/WRN 能力补齐（阶段6）

### 18.1 深度上限 8 → 24

- `net_spec.h` 的 `MAX_DEPTH` 提升到 **24**。所有以它为大小的静态数组
  （层数组、`off*/n*` 偏移表、`GpuBuffers` 缓冲池、`flattenParams`/`scatterParams` 的逐层循环）
  自动同步，无需手改。
- `NetSpec` 按值传入 CUDA kernel，其体积（MAX_DEPTH=24 时约 2.4KB）仍小于 kernel 参数
  4KB 上限，故**保持按值传参**（`net_spec.h` 末尾的 `static_assert` 锁死该前提）。
- depth≤3 的既有结构与数值 golden 逐位不变。

### 18.2 1×1 投影残差（`res=2`）

- 语义：`y = F(x) + Wproj * x_skip`，`Wproj` 为 **1×1 卷积**（stride=1/pad=0），
  用于**通道对齐**；空间尺寸仍须一致（源层最终输出与本层目标空间相同）。
- 参数布局：每层在 `K|B|[BN]` 之后追加 `PW(outC×srcC) | PB(outC)`，**仅 `res=2` 的层**非零；
  其余层数量为 0、偏移塌缩，故非投影网络的布局与 `total` 逐位不变。
- 文件格式（新 tag，旧程序读到 `res=2` 即干净报错）：
  ```
  R<i> 2 <resFrom>          # 投影残差
  RP<i> <outC> <srcC>       # 1×1 权重，随后 outC*srcC 个浮点
  PB<i> <outC>              # 偏置，随后 outC 个浮点
  ```
- 前向/反向（CPU `trainOne` + GPU `kproj_add`/`kproj_bwd`）：投影分支与主干并行，
  反向累计 `dPW/dPB` 并把 `Wproj^T * dy` 分流回源层。

### 18.3 pre-activation 残差位置（`resPos=1`）

- 在 `LayerSpec` 新增 `resPos` 字段（默认 0 = 旧语义）：
  - `resPos=0`：残差加在**本层最终输出（Pool 后）**——既有 `R<i>` 行为，逐位不变；
  - `resPos=1`：残差加在**本层 BN 之后、ReLU 之前**（标准 basic block 位置）：
    `b = bn(conv(x))；h = b + skip；y = pool(relu(h))`。
- 文件格式：`R<i> <res> <resFrom> [<resPos>]`，`resPos` **仅在非 0 时**追加，
  旧三字段文件不受影响；`resPos` 通过「下一字符是否为数字」探测读取。
- CPU/GPU 前向在 ReLU 前做 add 并把结果送入 ReLU；反向从 ReLU 前的梯度分流。

### 18.4 新增 CLI 与回归

- `gen --res2 --depth N` / `bkc ... --res2`：生成 1×1 投影残差网（末层 `res=2`）。
- `gen --preact --depth N` / `bkc ... --preact`：生成 pre-activation 残差网（末层 `resPos=1`）。
- `gen --res --depth 16`：生成 depth=16 的深残差网（验证深度上限）。
- `--gradcheck` 新增 `projres`（对拍 `proj_w`/`proj_b`）与 `preactres`（对拍 pre-activation 分支）两个网络类型。
- 回归新增：`gen_deep_struct` / `gen_proj_struct` / `gen_preact_struct`（结构）、
  `cpu_proj`/`gpu_proj`、`cpu_preact`/`gpu_preact`（固定夹具数值，CPU=GPU）、
  `gradcheck_proj`/`gradcheck_preact`（梯度）。原 34 条 golden 全部保留且全绿。

---

*文档维护提醒：修改任何命令行参数、模型格式、BN 规则、预处理默认值或构建配置后，
请同步更新本 README 对应章节。*
