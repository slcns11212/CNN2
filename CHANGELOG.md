# Changelog

本文件记录 CNN2 的主要变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

> **注意**：此处记录的「程序版本」与模型文件 `#SIG` 行中的「模型格式版本 `ver=`」
> 相互独立。程序升级不必改格式；一旦改动权重文件布局必须递增模型格式版本，详见
> README「版本与兼容」小节。

## [Unreleased]

### Added

- **终端 TUI 指令台（`cnn2 tui` / `--tui`）**：新增 opt-in 全屏 TUI，直接输入 CNN2 指令并回车执行，
  输出显示在内容区；带 MC 聊天栏式自动补全（Tab 循环、空格进入下一 token、↑↓ 历史、Esc 关闭候选）、
  内建 `help`/`clear`/`quit`/`cd`/`history`/`status`，子进程执行，Ctrl+C 中止/退出。
  默认仍是普通命令行交互，仅 `tui` / `--tui` 进入 TUI。
- **引入 FTXUI 7.0.3**：源码 vendored 到 `third_party/ftxui/`（含鼠标补丁），构建无需额外安装依赖。

- **模型文件系统革新（acm 二进制格式 / `--txtmodel` / `read` / 生成后缀）**：
  - **acm 二进制模型格式（默认保存格式）**：固定 **1024 字节头**（魔数 `ACMN` / 头版本 /
    格式版本 `ver=` / **精度字段 `precision`（占位，当前仅 fp32）** / flags / `#SIG` 原文 /
    24×20B 层表 / 保留区）+ fp32 数据区（`total` 个 float32，与 `flattenParams` 同序），
    文件大小恒为 `1024 + total*4`；acm **无损承载 float32 权重**，txt 为 **6 位有效数字
    的文本近似**（acm↔txt 互转时权重经 txt 会损失精度，并非逐位无损）。
    读侧恒按魔数自动识别（不受 `--txtmodel` 影响）；后缀 `.acm` 但魔数不符判为损坏；
    `precision != 0` 时 `read` / 加载均报「暂不支持」。新增 `file.h`（路径/格式探测 +
    acm 头编解码 + CRC32，header-only）。
  - **`--txtmodel`（含环境变量 `CNN2_TXTMODEL`）**：强制以 txt 文本格式保存模型；
    env 非空且非 `0` 时把默认保存格式切回 txt。未给开关时默认保存为 acm。
  - **文件名后缀规则**：`.txt`→txt / `.acm`→acm（`.acm` 与 `--txtmodel` 同用则报错退出）/
    无或其它后缀看开关（默认 acm）。**默认模型名改为 `model.acm`**（`--txtmodel` 时为
    `model.txt`）；每 10 轮的 `model_epochN`、`--epsave` 的 `ep_<epoch>`、`--val` 的
    `<stem>.best` 其后缀均随格式。
  - **新命令 `read`**：`cnn2 read <模型文件>` / `cnn2 read --model <模型文件>`，
    轻量查看模型（不训练 / 不推理），自动识别 acm/txt 并打印格式、`#SIG` 等价信息、
    维度与逐层结构、`FC` 行；成功退出 0，失败退出 1。
  - **回归适配（`build_and_test.ps1`）**：既有回归在脚本最前注入 `CNN2_TXTMODEL=1` 走 txt
    路径，原断言零改动继续通过；新增 acm 专项块（`-OnlyAcm` 可单独运行）：`acm_default`、
    `acm_suffix_acm`/`acm_suffix_txt`/`acm_suffix_conflict`、`acm_txt_equiv`、`acm_size`、
    `acm_precision`/`acm_precision_load`、`acm_read_smoke`。

- **L2 weight decay（`--wd <float>`，阶段7）**：coupled L2 语义（`g += wd*param`），默认 `0.0`。
  仅作用于**乘法权重**（卷积核 `K`、1×1 投影权 `PW`、FC 权 `FCW`），BN `γ/β` 与各段 `bias`
  **不衰减**。`--wd` 缺省或 `--wd 0` 时不进入叠加路径，训练/输出**逐位不变**（位保护）；
  CPU 与 GPU 两条训练路径均支持（`GpuTrainConfig.weightDecay` + `isWeightDecayParam`）。
- **每轮存档（`--epsave [目录]` / `--epsn <N>`，阶段7）**：`--epsave` 开启后 `bkc` 在每个满足
  `epoch % epsn == 0` 的 epoch 结束把权重保存为 `<目录>\ep_<epoch>.txt`（`epoch` 从 0 计）。
  目录省略时默认 `EPTSET\<模型名>\`（相对当前工作目录）；`--epsn` 为保存间隔，默认 1（每轮），
  非法值（`<1`）clamp 为 1 并在 stderr 警告。与既有「每 10 轮存 `model_epochN.txt`」并存、
  互不影响；**未传 `--epsave` 时行为零改变**。
- **回归新增**：`wd_bp`（`--wd 0` vs 缺省 SHA256 位保护）、`wd_active`/`wd_active_diff`
  （`--wd 0.01` exit=0 + loss 有限 + SHA256 与 `--wd 0` 不同）、`epsave`（`ep_0/ep_1` 生成且可被
  `test` 加载）、`epsn`（`--epsn 2` 仅生成 `ep_0/ep_2`）。`README` 参数总表新增三个参数条目，
  并给出 `1e-4 ~ 1e-3` 的 `--wd` 起步建议。

- **ResNet/WRN 能力补齐（阶段6）**：
  - **深度上限 8 → 24**：`MAX_DEPTH` 提升，所有以其为大小的静态数组自动同步；
    因 `NetSpec` 体积仍小于 CUDA kernel 参数 4KB 上限，保持按值传参。
  - **1×1 投影残差（`res=2`）**：`y = F(x) + Wproj*x_skip` 用于通道对齐；新增每层
    `PW(outC×srcC)|PB(outC)` 参数段（仅 `res=2` 非零）与新 tag
    `RP<i>`/`PB<i>`；CPU/GPU 前向/反向（含 `dPW/dPB` 与梯度分流）完整支持。
  - **pre-activation 残差（`resPos=1`）**：新增 `LayerSpec.resPos` 字段与可选第 4 字段，
    残差加在 BN 之后、ReLU 之前；`resPos` 缺省=0 即旧语义，旧三字段文件逐字节兼容。
  - **CLI**：`gen --res2`（投影）、`gen --preact`（pre-activation）、`gen --res --depth 16`（深度）。
  - **`--gradcheck`** 新增 `projres`（对拍 `proj_w`/`proj_b`）与 `preactres` 网络类型。
  - **回归新增**：`gen_deep_struct`/`gen_proj_struct`/`gen_preact_struct`、
    `cpu_proj`/`gpu_proj`、`cpu_preact`/`gpu_preact`、`gradcheck_proj`/`gradcheck_preact`；
    原 34 条 golden 全部保留且全绿。

- **一键发布与可复现构建（`release.ps1`，阶段4.5）**：新增项目根脚本 `release.ps1`，
  从 `version.h` 读版本 → MSBuild 编译 Release x64（带超时保护）→ 打包
  `dist\CNN2-<版本>-win-x64\`（`CNN2.exe` + 4 个运行时 dll + `README.md` +
  `CHANGELOG.md` + `BUILDINFO.txt`，记录程序/格式版本、构建日期、主机、工具链、
  exe SHA256）→ `Compress-Archive` 压缩为 `dist\CNN2-<版本>-win-x64.zip` 并打印 SHA256；
  运行时 dll 缺失时明确报错并列出缺少项。README 新增「12.3 发布 / 可复现构建」小节。
  本脚本不改核心源码，也不改 `build_and_test.ps1`（原回归用例不受影响）。
- **独立验证集（`--val <目录>`，阶段4.4）**：默认关闭。开启后 `bkc` 每 epoch 结束在验证集上
  评估一次准确率，预处理/推理与 `test` 完全同源（`forwardConv` + `fc.forward` + softmax），
  不做增强、只读、不污染训练。每 epoch 打印 `val acc`；配合 `--log` 时日志追加 `val_acc`
  列（8 列），并把 val 最优模型存到 `<model>.best`（记一行 CHECKPOINT）。`val=train` 时告警。
  未传 `--val` 时训练 / stdout / 日志 / 存档逐字节不变。`build_and_test.ps1` 新增 `val` 用例。
- **数据校验（`--checkdata` / `--strict-data`，阶段4.4）**：`cnn2 <目录> --checkdata` 报告
  文件数、每类计数、非法文件清单（首字符非数字、空文件、非 32×32、含非 `0/1` 字符）；
  `--strict-data` 遇非法即报错并非零退出，也可用于 `bkc` 启动前校验训练/`--val` 目录。
  `build_and_test.ps1` 新增 `data_check` / `data_strict` 用例。

- **训练可观测性（`--log <file>`，阶段4.3）**：默认关闭。开启后向日志文件追加
  每 epoch 一行 CSV 指标（`epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec`，
  新文件先写表头）与每次存档一行 `CHECKPOINT <epoch> <path> <size> <time>` 元数据；
  CPU/GPU 两条训练路径均支持，追加 + 逐条 `flush` 保证崩溃留痕。未传 `--log` 时
  不创建文件、输出不变；GUI/test/推理无副作用。`build_and_test.ps1` 新增
  `train_log` 用例覆盖。

### Changed（本版重构）

- **CLI 归一化与子命令（阶段B）**：
  - **键名归一化**：`--` 后键名大小写不敏感、`-` 与 `_` 等价（如 `--lr-fc` ≡ `--lr_fc`、
    `--strict_data` ≡ `--strict-data`）；`--bn` 归并为内部键 `BN`。
  - **子命令式**：新增 `cnn2 train <目录>` / `cnn2 test <目录>` / `cnn2 gen <路径>` /
    `cnn2 infer <图>` / `cnn2 gui`，与旧式 `cnn2 <路径> <模式>` 并存。判别规则：仅当
    `argv[1]` 为子命令词且 `argv[2]` 非模式词（`gen`/`bkc`/`test`）时才按子命令解析，
    否则沿用旧式（`cnn2 train bkc` 仍为路径 `train`、模式 `bkc`）。
  - **未知参数警告 + did-you-mean**：未知 `--key` 打印警告并忽略（必要时给出最近似建议），不退出。
  - **`--out <路径>`**：`gen` 模式作为 `--model` 别名（未显式给 `--model` 时应用）。
- **ODR 修复**：`conv.h` / `sf.h` 的全局函数/变量 `inline` 化，消除多编译单元重复定义隐患；
  并把 `conv.h` 由 GBK 转为 **UTF-8 with BOM**，统一源码编码规范。
- **常量收敛**：新增 `common.h`（`CNN_PI`/`CNN_BN_EPS`/`CNN_LOSS_EPS`/`CNN_GOLDEN_RATIO`
  及 `argmaxIndex`/`formatDur`），作为跨文件共享常量与工具的**唯一来源**，消除重复定义漂移。

### Fixed（本版重构）

- **GUI**：多模型路径给出明确报错（不再静默忽略）；忽略未知参数并打印警告；
  加载后 `checkModelDims` 校验模型形状与 `NetSpec` 自洽（避免 flatten/scatter 越界）。
- **诊断能力**：`parseSigFields` 恢复对非法 `ver=` 字段的 stderr 警告（仅诊断，不再存回 `NetSpec`）。
- **子命令解析**：修正与子命令同名的路径被误判（`cnn2 train bkc` 等恢复为旧式解析）。

### Removed（本版重构）

- **死代码精简**：移除旧参数别名、死宏、`makeNetSpec`、`NetSpec::ver` 字段、
  `outProbSum`、GUI 按钮全局变量等不再使用的代码。

## [1.0.0] - 2026-10-04

首个正式版本。汇总截至当前的架构、训练、评估与工程化里程碑。

### Added

- **架构可配置化**：新增 `--c1` / `--c2` 指定 Conv1/Conv2 输出通道；网络维度由运行时
  `NetSpec`（`net_spec.h`）统一推导，主机与 GPU 共用同一份规格与参数偏移，
  彻底消除硬编码偏移。
- **NetSpec 层数组**：`NetSpec` 由「写死两层」重构为「层列表 + 逐层偏移/数量」，
  为可变深度与残差奠定基础；默认 32/64 两层布局与改造前逐位一致。
- **可选 BatchNorm（`--BN` / `--bn`）**：真 BatchNorm 前向/反向/统计量更新（`γ`/`β`
  可训练，`running_mean`/`running_var` 为统计量）；GPU 训练支持 BN，CPU/GPU/test/
  单图/GUI 支持 BN 推理；`BN:ON` 模型必须显式传 `--BN`。
- **One-Cycle 学习率调度（`--onecycle`）**：warmup 线性升 + 余弦退火；默认关闭时与固定
  LR 逐位一致。配套 `--lr_warmup` / `--lr_min` / `--lr_peak`。
- **集成与融合（test）**：`--model a.txt,b.txt,...` 支持多模型集成；
  `--fuse soft|logit|geom|vote|gate|wbest`、`--weights`、`--fuse_margin`、`--fuse_temp`。
- **TTA（测试时增强）**：`--tta` / `--tta_k`，对已预处理图做固定几何变换序列，
  在 log 概率空间平均。
- **判错清单（`--outerror`）**：test 时输出判错样本（屏幕 + `outerror.txt`），
  含真值、预测、置信度与真值概率。
- **模型签名 `#SIG` 与 `--P`**：模型头部强制写签名行，记录 `BN:ON|OFF`、`depth`，
  `--P` 时可附带 GPU 名/计算机名/时间；加载时解析签名并归一化旧格式。
- **可变深度（`--depth`）与 `P<i>` / `#SIG ver=2`**：`depth>2` 使用默认同构
  conv→pool schedule；每层可写显式 `P<i> <stride> <pad> <pool>`；`depth>2` 或存在
  非默认 `P<i>` 时签名追加 `ver=2`。
- **残差连接（`--res`，identity skip）**：新增 `R<i> <res> <resFrom>` 格式，支持
  ResNet/WRN basic block 语义；CPU/GPU 前向与反向（梯度分流）均支持。
- **可复现训练（`--seed`）**：初始化、采样、每 epoch 洗牌、数据增强全部确定化；
  同 seed + 同数据 + 同参数下 `gen`/`bkc` 产物与日志逐位一致。
- **安全加固**：模型加载全流程 `try/catch`（`bad_alloc` / `length_error`），
  维度与容量校验前置，坏模型/超大分配不再 `terminate`。
- **并行数据加载**：GPU 路径 `--threads`（默认 8，0=auto）多线程读图 + 预处理 + 离线增强；
  线程本地缓冲 + 无锁合并，与线程划分无关地保证可复现。
- **CPU 并行推理**：test CPU 路径按文件区间多线程读图/预处理，样本顺序与串行一致。

### Added（工程化）

- **版本化能力**：新增 `version.h` 集中定义程序版本 `CNN2_VERSION` 与模型格式版本
  `CNN2_MODEL_VER`，并新增 `--version`（打印程序版本、模型格式版本、构建日期/时间，
  打印后退出 0，不加载数据/不训练）。
- **`build_and_test.ps1` 回归套件**：编译 Release x64 + 逐条 golden 断言
  （推理数值、gen 结构、报错退出码、训练冒烟、可复现 SHA256、梯度校验），
  新增 `--version` 用例。
- 新增本 `CHANGELOG.md`；README 增加「版本与兼容」小节。

### Notes

- 模型格式版本 `ver=1`（含无 `#SIG` 最老文件）保持向后兼容读取；
  当前格式版本为 `ver=2`（可变深度 / `P<i>` / 残差）。
- `conv.h` / `sf.h` 保持旧版 GBK 编码，未改动。
