# CNN2 性能基准（BENCH）

本文件说明如何用**现有 Release 二进制**做可复跑的性能基准、各指标含义、本次基线数值，以及改动后如何对比。
基准脚本为项目根目录的 `bench.ps1`；它**不修改任何核心源码**，只用现有 `--time`/stdout 输出做解析。

---

## 1. 如何跑

前置：已编译 `x64\Release\CNN2.exe`（脚本会检测，缺失即报错退出）。

```powershell
# 默认：seed=12345、epochs=2、含 CPU 训练与全部推理
.\bench.ps1

# 日常/CI 推荐：跳过很慢的 CPU 训练（只跑 GPU 训练 + CPU/GPU 推理）
.\bench.ps1 -Epochs 2 -SkipCpuTrain

# 更快：1 个 epoch
.\bench.ps1 -Epochs 1 -SkipCpuTrain

# 无 GPU 环境：GPU 相关项自动记 Skip
.\bench.ps1 -SkipGpu

# 其它开关
.\bench.ps1 -Seed 12345 -TimeoutSec 1800 -OutDir bench_results
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Seed` | `12345` | 固定随机种子（初始化 / 每 epoch 洗牌 / 增强），保证可复现 |
| `-Epochs` | `2` | 训练轮数（基准量级，**不做正式训练**） |
| `-SkipCpuTrain` | off | 跳过 CPU 训练（BN 配置无论如何只做 GPU 训练） |
| `-SkipGpu` | off | 无 GPU 环境；GPU 训练/推理记 SKIP |
| `-TimeoutSec` | `1800` | 单个 exe 进程的超时秒数，超时树杀并按 FAIL 处理 |
| `-OutDir` | `bench_results` | 报告输出目录（相对项目根） |

产物：

- `bench_results\bench_<yyyyMMdd_HHmmss>.md`：本次报告；
- `bench_results\bench_latest.md`：最新报告副本（便于脚本化对比）；
- `bench_results\models\*.txt`：本次训练产物（同配置推理使用）；
- 报告同时打印到 stdout。

超时保护：每个 exe 以独立进程启动（`cmd /c` + 重定向），`Process.WaitForExit(TimeoutSec*1000)`；超时即 `taskkill /F /T /PID` 树杀，行内记 `FAIL / 超时(已 Kill, Ns)`。与 `build_and_test.ps1` 的写法一致。

覆盖配置：

- 训练：默认 `depth2`、`--depth 3`、`--res --depth 3`、`--BN`（GPU 全跑；BN 仅 GPU 可训练，CPU 训练会 Skip）。
- 推理：每个配置的 **CPU + GPU**（使用本次训练出的同配置模型）。
- 跑不了的项在表内标 `SKIP` 并注明原因（如 `GPU 不可用`、`-SkipCpuTrain`、`CPU 训练不支持 BN`）。

数据路径（脚本工作目录固定为 `x64\Release`）：

- 训练：`trainingDigits` → `x64\Release\trainingDigits`（1934 张）
- 测试：`..\testDigits` → `x64\testDigits`（10000 张）

---

## 2. 指标口径

| 指标 | 口径 |
| --- | --- |
| 数据加载(s) | `--time` 报告的加载耗时（GPU：`数据加载耗时`；CPU：`数据加载累计`） |
| epoch墙钟(s/epoch) | 各 epoch `耗时` 的平均值（CPU 含该 epoch 在线加载；GPU 不含加载） |
| 训练吞吐(样本/s) | 样本数 ÷ 计算时长（CPU 取 `计算` 明细，GPU 取 epoch 耗时） |
| 显存(MB) | 程序 `预计显存占用 … ≈ X MB`（仅 GPU 训练打印） |
| 推理吞吐(样本/s) | 测试文件数(10000) ÷ 进程端到端墙钟（含启动 / 加载模型 / 预处理 / 推理） |
| 准确率 | `标准准确率 (correct/total)`，**仅做 sanity**；2 epoch 严重欠拟合，不用于质量评估 |

> CPU 训练说明：CPU 路径**在线增强**、不预扩增，且不打印“扩增后 N 样本”。
> 脚本对 CPU 训练用**训练集文件数**作为每 epoch 样本数回退（备注 `CPU 样本数=训练集文件数`），使 CPU 吞吐可计算。

---

## 3. 本次基线（快照）

- 生成时间：**2026-10-04 11:45:26**，脚本总耗时 **233.1 秒**
- 命令：`.\bench.ps1 -Epochs 2 -SkipCpuTrain`
- seed=12345，epochs=2
- exe SHA256（前 16）：`76CFE2C2993D2D37`，大小 607,232 字节
- 快照文件：`bench_results\bench_20261004_114133.md`（另存 `bench_results\bench_baseline.md`）

### 3.1 训练

| 配置 | 后端 | 加载(s) | epoch墙钟(s/epoch) | 训练吞吐(样本/s) | 总训练(s) | 显存(MB) | 状态 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| depth2 | GPU | 0.09 | 0.51 | 3,792 | 1.26 | 67.2 | OK | |
| depth2 | CPU | - | - | - | - | - | SKIP | -SkipCpuTrain |
| depth3 | GPU | 0.10 | 0.99 | 1,963 | 2.23 | 99.8 | OK | |
| depth3 | CPU | - | - | - | - | - | SKIP | -SkipCpuTrain |
| res | GPU | 0.11 | 0.72 | 2,705 | 1.74 | 105.2 | OK | |
| res | CPU | - | - | - | - | - | SKIP | -SkipCpuTrain |
| BN | GPU | 0.10 | 0.50 | 3,868 | 1.24 | 100.7 | OK | |
| BN | CPU | - | - | - | - | - | SKIP | -SkipCpuTrain |

### 3.2 推理（10000 样本）

| 配置 | 后端 | 墙钟(s) | 吞吐(样本/s) | 准确率 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| depth2 | CPU | 37.08 | 270 | 0.2321 | OK |
| depth2 | GPU | 3.07 | 3,259 | 0.2321 | OK |
| depth3 | CPU | 78.07 | 128 | 0.2627 | OK |
| depth3 | GPU | 4.22 | 2,367 | 0.2627 | OK |
| res | CPU | 55.31 | 181 | 0.2277 | OK |
| res | GPU | 3.68 | 2,716 | 0.2277 | OK |
| BN | CPU | 39.69 | 252 | 0.2628 | OK |
| BN | GPU | 3.21 | 3,116 | 0.2628 | OK |

**基线要点**：GPU 训练吞吐 1.9k–3.9k 样本/s，显存 67–105 MB；GPU 推理吞吐 2.4k–3.3k 样本/s，
约为 CPU 推理（128–270 样本/s）的 **约 10–25 倍**。深度越深越慢（depth3 最慢），`res` 因残差额外开销居中。

---

## 4. 环境信息

| 项 | 值 |
| --- | --- |
| 主机 | GLDERETOP-1OGFE |
| 操作系统 | Windows 10 专业版（10.0） |
| CPU | Intel(R) Core(TM) i7-7820HK @ 2.90GHz，4 物理核 / 8 逻辑核 |
| 内存 | 23.9 GB |
| GPU | NVIDIA GeForce GTX 1070，8192 MiB |
| GPU 驱动 | 582.28 |
| 编译器 | MSVC v143（`cl` 19.44.35208，工具集 14.44.35207） |
| C++ 标准 | C++17（`stdcpp17`） |
| CUDA Toolkit | 12.9.86（Runtime: Shared） |
| Windows SDK | 10.0 |
| exe | SHA256 前 16 位 `76CFE2C2993D2D37`，607,232 字节 |

---

## 5. 改动后如何对比

1. **固定环境**：尽量在同一台机器、同一 exe 数据/依赖下比较；脚本会记录 CPU/GPU/CUDA/编译器与 exe SHA256。
2. **先存基线**：把当前 `bench_latest.md` 另存（本仓库已存 `bench_results\bench_baseline.md`）；或直接引用带时间戳的历史报告。
3. **改动后重跑**：用**完全相同的参数**，例如
   `.\bench.ps1 -Epochs 2 -SkipCpuTrain`。
4. **比较**：
   - 逐行看 `bench_latest.md` 的加载 / epoch墙钟 / 吞吐 / 显存，对比同配置同后端；
   - 吞吐取整数量级，不要对单次墙钟做小数级强判（GPU 训练总时长仅 1–2 秒，噪声占比高）；
   - 确认 `exe SHA256` 已变化（否则对比的是同一二进制）；
   - 准确率列仅确认“能跑通、数值合理”，不做质量判定。
5. **噪声控制**：
   - 推理吞吐含进程启动、模型加载与预处理，属端到端口径；关注**相对倍数**更稳；
   - **不要与其他 bench/训练并发**：CPU 推理与磁盘 IO 会互相抢占，导致 CPU 项明显变慢；
   - 建议重复 2–3 次取中位数，尤其 CPU 推理与 `res` 配置。

---

## 6. CPU 训练（可选，很慢）

去掉 `-SkipCpuTrain` 即可测 CPU 训练，但会很慢（残差配置尤甚，量级可达数分钟/epoch）。
该路径用于验证 CPU 训练正确性/吞吐趋势，不适合每次跑。基线默认跳过并在表中记 `SKIP / -SkipCpuTrain`。

---

## 7. bench.ps1 结构（便于维护）

| 函数 / 段 | 职责 |
| --- | --- |
| `Invoke-Exe` | 独立进程 + 超时树杀；以 GBK(936) 捕获 stdout/stderr，返回 `Exit/Text/TimedOut/WallSec` |
| `ConvertTo-Seconds` | 把 `1.23 秒` / `1 分 23.45 秒` 转成秒 |
| `Test-GpuUnavailable` | 识别 `no CUDA device available`，一次失败后全局 `gpuOk=$false` |
| `Parse-Train` | 从 `--time` 输出解析加载/总时长/样本数/每 epoch 墙钟与计算/显存 |
| `Invoke-TrainBench` | 跑一次训练，组装训练行（含 CPU 样本数回退） |
| `Invoke-InferBench` | 跑一次 `test --cpu/--gpu`，解析文件数/准确率/吞吐 |
| `Get-EnvInfo` | 采集主机/OS/CPU/内存/GPU/CUDA/编译器/exe/数据规模 |
| 主流程 | 遍历 4 配置 × {GPU 训练, CPU 训练, CPU/GPU 推理}，生成并写出 Markdown 报告 |

---

## 8. 已知风险 / 说明

- **不改核心源码**：基准完全依赖现有 `--time` 与 stdout 文本；若程序输出文案变化，`Parse-Train`/`Invoke-InferBench` 的正则需同步更新。
- **并发写风险**：多个 bench/构建 job 同时运行会争抢同一数据目录、`bench_results\` 与 `x64\Release\` 下的模型文件，结果可能失真；建议串行执行，落定后复跑确认。
- **路径依赖**：训练/测试数据按 `x64\Release\trainingDigits` 与 `..\testDigits`（= `x64\testDigits`）解析，移动 exe 或数据需同步调整。
- **准确率非目标**：2 epoch 下准确率低是预期现象，仅用于确认流程可复现。
- 本文件与 `bench.ps1` 均为阶段5新增/调整，**未修改** `build_and_test.ps1`、`conv.h`、`sf.h` 等核心源码。
