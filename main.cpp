#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>          // 跨平台计时（steady_clock，Windows/Linux 通用）
#include <windows.h>
#include <algorithm>
#include <random>
#include <map>
#include <climits>
#include <cctype>
#include <new>
#include <stdexcept>       // std::length_error（内存分配健壮性）
#include <cstdio>          // snprintf（跨平台时间格式化）
#include <thread>          // 多线程数据加载（--threads）
#include <atomic>          // 线程安全进度计数 / diagPrinted
#include <mutex>           // 进度打印互斥

#include "conv.h"
#include "sf.h"
#include "preprocessing.h"
#include "augment.h"
#include "gpu_trainer.cuh"
#include "GUI.h"
#include "net_spec.h"
#include "file.h"        // T1/T2：acm 二进制格式编解码 + 路径/格式探测（header-only）
#include "version.h"
#include "common.h"      // 共享常量（PI/BN_EPS/LOSS_EPS/GOLDEN_RATIO）与 formatDur
#include "TuiLauncher.h"   // 终端 TUI 模式（cnn2 tui / --tui）
#include "tui/TuiHelpText.h"  // TUI 帮助文案单一来源（与 TUI 内建 help 一致）

// ================= 全局模型（阶段1：层数组） =================
// 主机侧权重按层存放；下标为 0-based 层号（本阶段 depth==2，仅用 [0]/[1]）。
using K4 = std::vector<std::vector<std::vector<std::vector<float>>>>;
K4 KERNELS_L[MAX_DEPTH];                                // 每层卷积核 [outC][inC][kH][kW]
std::vector<float> BIAS_L[MAX_DEPTH];                   // 每层偏置 [outC]

// ================= BN 张量（P1：主机侧权重 + 统计量） =================
// γ/β 为可训练权重（参与反向/更新）；
// running_mean/running_var 为跨 batch 统计量：训练时更新、推理时用于归一化。
// 按层存放；g_net.bn=false 时不使用（保持为空）。
std::vector<float> BN_G_L[MAX_DEPTH], BN_B_L[MAX_DEPTH];
std::vector<float> BN_MEAN_L[MAX_DEPTH], BN_VAR_L[MAX_DEPTH];

// ================= 1×1 投影残差权重（阶段6，仅 res==2 的层非空） =================
// 形状 [outC][srcC][1][1]（1×1 卷积，stride=1/pad=0，用于通道对齐）；可为空表示该层无投影。
K4 PROJ_K_L[MAX_DEPTH];
std::vector<float> PROJ_B_L[MAX_DEPTH];

// ================= 全局预处理配置（训练/测试/推理共用） =================
static PrepConfig g_prep;

// ================= 运行时网络规格（阶段0+1：主机侧可配置） =================
// 默认 32/64，与改造前编译期常量一致；加载模型/生成模型前由 peek/gen 决定。
static NetSpec g_net;

// ================= 可复现种子（阶段4.1） =================
// --seed 指定时：初始化 / 每 epoch 洗牌 / 增强全部由它（经 mixSeed 派生子种子）决定；
// 未指定（g_seedSet=false）时各随机源保留时间/系统随机，默认随机行为不变。
static unsigned g_seed = 0;
static bool     g_seedSet = false;

// ================= GUI 模式（--gui）使用的全连接层指针 =================
static FullyConnected* g_guiFc = nullptr;

// ================= 模型签名（--P） =================
static bool        g_modelSign = false;  // --P：saveModel 是否写完整签名（设备/主机/时间）
static std::string g_loadedSig;          // loadModel 读到的原签名行（无则为空）
static bool        g_useBN = false;      // --BN/--bn：是否请求 BN；最终以 g_net.bn 为准
static bool        g_useRes = false;     // --res：是否请求残差网络（阶段3；gen/bkc 新建时生效）
static bool        g_saveAcm = true;     // T5：保存格式开关（默认 acm；--txtmodel / CNN2_TXTMODEL 置 false）

// ================= 训练日志（--log，阶段4.3 可观测性） =================
// 开启后：bkc 每 epoch 追加一行 CSV 指标；每次 saveModel 追加一行 CHECKPOINT 元数据。
// 未传 --log 时 g_logEnabled=false，下列函数立即返回，默认行为与输出完全不变。
// 日志文件为纯 ASCII CSV（无 BOM），追加模式且每条 flush，进程崩溃也能留痕。
static bool          g_logEnabled = false;
static std::ofstream g_log;
static const char*   kLogHeader =
    "epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec\n";

// ================= 独立验证集（--val，阶段4.4） =================
// --val 非空时：bkc 每 epoch 结束后用 CPU 前向（forwardConv + fc）在验证集上算一次准确率，
// 预处理与 test 同源（loadImagePreprocessed(g_prep)），且不做增强。
// 未传 --val 时 g_valEnabled=false：训练流程 / stdout / 日志列与改动前逐字节一致。
// 开启 --val 后日志追加 val_acc 列（1 行表头也随之为 8 列），从而消除「用 test 调参」的评估污染。
static bool               g_valEnabled = false;
static std::string        g_valDir;
static std::vector<float> g_valData;    // N*1024 连续 0/1 输入（与 test/GPU 路径一致）
static std::vector<int>   g_valLabels;  // N
static float              g_bestValAcc = -1.0f; // 历史最优 val 准确率（<0 表示尚无）
static bool               g_saveBest = false;   // --val 时默认把最优模型存到 <model>.best
static std::string        g_bestPath;           // <model>.best 路径
static const char* kLogHeaderVal =
    "epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec,val_acc\n";

// ================= 每轮存档（--epsave / --epsn） =================
// --epsave [目录]：开启「每轮保存权重」。目录省略时默认 EPTSET\<模型名>\（相对 CWD）。
// --epsn N       ：保存间隔（默认 1=每轮），仅 --epsave 开启时生效。
// 未传 --epsave 时 g_epsaveEnabled=false：训练流程 / stdout / 目录均不变（默认行为零改变）。
static bool        g_epsaveEnabled = false;
static std::string g_epSaveDir;   // 末尾带 '\'
static int         g_epsn = 1;

// 写一行 epoch 指标（追加 + 立即 flush）。epoch_sec 为本 epoch 墙上耗时（秒）。
// hasVal 仅在 g_valEnabled 时参与：追加 val_acc 列（无有效验证样本时写 -1）。
// 未开启 --val 时字段数与字节输出与改动前完全一致（保持 7 列）。
static void logEpochMetrics(int epoch, float loss, float acc,
                            float lr1, float lr2, float lrfc, double epochSec,
                            float valAcc = -1.0f, bool hasVal = false) {
    if (!g_logEnabled) return;
    g_log << epoch << "," << loss << "," << acc << ","
          << lr1 << "," << lr2 << "," << lrfc << "," << epochSec;
    if (g_valEnabled) g_log << "," << (hasVal ? valAcc : -1.0f);
    g_log << "\n";
    g_log.flush();
}

// 写一行 checkpoint 元数据：CHECKPOINT <epoch> <path> <size_bytes> <time>
// 用「文件大小 + 保存时刻」作为最小元数据（避免每次存档重算 sha256 的开销）；
// 文件不可读时 size 记为 -1。使「模型文件 ↔ 指标 ↔ 时间」可追溯。
static void logCheckpoint(int epoch, const std::string& path) {
    if (!g_logEnabled) return;
    long long size = -1;
    {
        std::ifstream fin(path, std::ios::binary | std::ios::ate);
        if (fin) { size = (long long)fin.tellg(); fin.close(); }
    }
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_s(&tmv, &t);
    char ts[32] = { 0 };
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tmv);
    g_log << "CHECKPOINT " << epoch << " " << path << " " << size << " " << ts << "\n";
    g_log.flush();
}

// 计算机名（ANSI/本地代码页）；失败返回 "N/A"
static std::string getHostName() {
    char buf[256] = { 0 };
    DWORD n = (DWORD)sizeof(buf);
    if (!GetComputerNameA(buf, &n) || n == 0) return "N/A";
    return std::string(buf);
}

// ---- 签名字段辅助（P0） ----
static inline std::string sigTrim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}
static inline std::string sigLower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
// 去掉 token 可能的前导 "#SIG"（大小写不敏感），便于逐字段解析
static inline std::string sigStripHead(const std::string& tok) {
    std::string t = sigTrim(tok);
    if (!t.empty() && t[0] == '#') {
        t = sigTrim(t.substr(1));
        if (t.size() >= 3 &&
            (t[0] == 'S' || t[0] == 's') &&
            (t[1] == 'I' || t[1] == 'i') &&
            (t[2] == 'G' || t[2] == 'g'))
            t = sigTrim(t.substr(3));
    }
    return t;
}

// 组装签名行（头部强制写；GPU 名可能含空格，字段用 " | " 分隔，前缀用 '#'）：
//   withDevice=true （--P）: "#SIG gpu=.. | host=.. | time=.. | BN:ON|OFF | depth=2 [| ver=2]"
//   withDevice=false      : "#SIG BN:ON|OFF | depth=2 [| ver=2]"
// ver: 1=旧格式（无 P<i>）；2=depth>2 或存在非默认 P<i> 时追加 " | ver=2"。
static std::string makeSignature(bool bn, int depth, bool withDevice, int ver = 1) {
    const std::string bnPart =
        std::string("BN:") + (bn ? "ON" : "OFF") + " | depth=" + std::to_string(depth);
    std::string out = "#SIG";
    if (withDevice) {
        std::time_t t = std::time(nullptr);
        std::tm tmv{};
        localtime_s(&tmv, &t);
        char ts[32] = { 0 };
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tmv);
        out += " gpu=" + gpuName() + " | host=" + getHostName() + " | time=" + std::string(ts);
        out += " | " + bnPart;
    } else {
        out += " " + bnPart;
    }
    if (ver >= 2) out += " | ver=2";
    return out;
}

// 续训时不再原样复用旧签名字符串：保留旧签名中的设备/时间等字段（若有），
// 强制刷新 BN:/depth=/ver 字段。旧签名无设备字段则生成「仅 BN+depth[+ver]」格式。
static std::string normalizeSignature(const std::string& oldSig, bool bn, int depth, int ver = 1) {
    std::vector<std::string> keep;
    size_t pos = 0;
    while (pos <= oldSig.size()) {
        size_t bar = oldSig.find('|', pos);
        std::string tok = (bar == std::string::npos)
            ? oldSig.substr(pos) : oldSig.substr(pos, bar - pos);
        tok = sigStripHead(tok);
        if (!tok.empty()) {
            std::string low = sigLower(tok);
            bool isBn    = (low.rfind("bn:", 0) == 0 || low.rfind("bn=", 0) == 0);
            bool isDepth = (low.rfind("depth:", 0) == 0 || low.rfind("depth=", 0) == 0);
            bool isVer   = (low.rfind("ver:", 0) == 0 || low.rfind("ver=", 0) == 0);
            if (!isBn && !isDepth && !isVer) keep.push_back(tok);
        }
        if (bar == std::string::npos) break;
        pos = bar + 1;
    }
    const std::string bnPart =
        std::string("BN:") + (bn ? "ON" : "OFF") + " | depth=" + std::to_string(depth);
    std::string out = "#SIG";
    if (!keep.empty()) {
        for (size_t i = 0; i < keep.size(); ++i)
            out += (i ? " | " : " ") + keep[i];
        out += " | " + bnPart;
    } else {
        out += " " + bnPart;
    }
    if (ver >= 2) out += " | ver=2";
    return out;
}

// 解析签名行中的 BN / depth：按 '|' 切分、trim。
//   BN:/BN=（大小写不敏感）→ ON/OFF；非法 → OFF + 警告
//   depth=/depth: → int；非法 → 2
//   ver=/ver:     → int；非法 → 警告（仅诊断，不再存回 NetSpec，字段已删除）
// 缺省（含旧签名没有 BN: 字段）：BN=false、depth=2。
static void parseSigFields(const std::string& line, bool& bn, int& depth) {
    bn = false; depth = 2;
    size_t pos = 0;
    while (pos <= line.size()) {
        size_t bar = line.find('|', pos);
        std::string tok = (bar == std::string::npos)
            ? line.substr(pos) : line.substr(pos, bar - pos);
        tok = sigStripHead(tok);
        if (!tok.empty()) {
            std::string low = sigLower(tok);
            if (low.rfind("bn:", 0) == 0 || low.rfind("bn=", 0) == 0) {
                std::string v = sigTrim(low.substr(3));
                if (v == "on") bn = true;
                else if (v == "off") bn = false;
                else {
                    std::cerr << "警告: 非法 BN 字段 \"" << tok << "\"，按 BN:OFF 处理\n";
                    bn = false;
                }
            } else if (low.rfind("depth:", 0) == 0 || low.rfind("depth=", 0) == 0) {
                std::string v = sigTrim(low.substr(6));
                try {
                    depth = std::stoi(v);
                } catch (...) {
                    std::cerr << "警告: 非法 depth 字段 \"" << tok << "\"，按 depth=2 处理\n";
                    depth = 2;
                }
            } else if (low.rfind("ver:", 0) == 0 || low.rfind("ver=", 0) == 0) {
                // F8：恢复非法 ver 字段诊断。NetSpec::ver 已删除，此处仅做解析与
                // 合法性检查（不存回任何字段），非法时仍打印原警告，保持诊断能力。
                std::string v = sigTrim(low.substr(4));
                try {
                    int ver = std::stoi(v);
                    if (ver < 1) ver = 1;   // 与旧行为一致：低于 1 静默纠正（不告警）
                    (void)ver;              // 仅校验，不消费
                } catch (...) {
                    std::cerr << "警告: 非法 ver 字段 \"" << tok << "\"，按 ver=1 处理\n";
                }
            }
        }
        if (bar == std::string::npos) break;
        pos = bar + 1;
    }
}

// 判定某层 P 是否为默认 (stride,pad,pool)=(1,0,2)；默认层不写 P<i>。
static inline bool isDefaultP(const LayerSpec& L) {
    return L.stride == 1 && L.pad == 0 && L.pool == 2;
}

// ================= 已知参数键集合（阶段B：CLI 归一化/校验的唯一真源） =================
// 集中登记全部内部键名（不含前导 --）。新增参数时在此登记即可自动被
// B1 键名归一化与 B2 未知参数校验覆盖。内部键名保持现状不变，
// 故 "BN"/"bn"/"P"/"strict-data" 等历史写法全部在此登记。
static const std::vector<std::string>& knownArgKeys() {
    static const std::vector<std::string> keys = {
        // 训练 / 通用
        "epochs", "lr_fc", "lr_conv1", "lr_conv2", "momentum", "nesterov",
        "model", "batch", "gpu", "cpu", "time", "log", "val", "strict-data",
        "threads", "outerror", "seed", "data2", "mix", "full",
        "wd", "epsave", "epsn", "txtmodel",
        // 网络结构
        "P", "BN", "bn", "res", "res2", "preact", "c1", "c2", "depth",
        "no_conv1", "no_conv2", "no_fc",
        // One-Cycle 学习率调度
        "onecycle", "lr_warmup", "lr_min", "lr_peak",
        // test 集成 / TTA
        "fuse", "fuse_margin", "fuse_temp", "weights", "tta", "tta_k",
        // 预处理
        "prep", "prep_target", "prep_stretch", "prep_ink", "prep_deskew_max", "diag",
        // 数据增强
        "aug", "aug_k", "aug_shift", "aug_rotate", "aug_scale",
        "aug_elastic", "aug_elastic_sigma", "aug_thick",
        // 其它模式 / 工具
        "gc_samples", "gc_eps", "fpng", "gui", "tui", "version", "gradcheck", "checkdata", "cnn2", "help",
        "read",
        // B6：gen 模式 --out 等价 --model
        "out",
    };
    return keys;
}

// 把 knownArgKeys() 暴露为纯 C 视图，供 TUI 补全使用（单一真源，避免补全词典与 CLI 失步）。
// 返回的指针指向函数内静态存储，进程生命周期内有效；调用方不应保存超过进程生命周期。
static const char* const* cnn2TuiKnownArgKeys(int* count) {
    static const std::vector<const char*> ptrs = [] {
        std::vector<const char*> v;
        v.reserve(knownArgKeys().size());
        for (const std::string& k : knownArgKeys()) v.push_back(k.c_str());
        return v;
    }();
    if (count) *count = static_cast<int>(ptrs.size());
    return ptrs.data();
}

// 键名规范形式（B1）：小写 + '-' 与 '_' 等价，用于比较匹配。
static std::string canonicalArgKey(const std::string& s) {
    std::string r = s;
    for (char& c : r) {
        if (c == '-') c = '_';
        else          c = (char)std::tolower((unsigned char)c);
    }
    return r;
}

// 已知键「规范形式 -> 内部键名」映射（懒初始化，仅一次）。
static const std::map<std::string, std::string>& knownArgKeyMap() {
    static const std::map<std::string, std::string> m = [] {
        std::map<std::string, std::string> t;
        for (const std::string& k : knownArgKeys()) {
            // "bn" 与 "BN" 规范形式相同，统一归并到内部 "BN"
            // （下游均为 BN/bn 一起判读，语义等价）
            if (k == "bn") continue;
            t[canonicalArgKey(k)] = k;
        }
        return t;
    }();
    return m;
}

// B1：把用户键名映射为内部键名；返回空串表示归一化后仍未知。
static std::string normalizeArgKey(const std::string& raw) {
    // 1) 精确命中内部键（保留 BN/P/strict-data 等原样写法）
    for (const std::string& k : knownArgKeys()) {
        if (k == raw) return (raw == "bn") ? std::string("BN") : raw;
    }
    // 2) 别名表（特例）
    if (raw == "sign") return "P";   // --sign ≡ --P
    // 3) 规范形式匹配（大小写不敏感 + _/- 等价）
    const std::map<std::string, std::string>& m = knownArgKeyMap();
    std::map<std::string, std::string>::const_iterator it = m.find(canonicalArgKey(raw));
    if (it != m.end()) return it->second;
    return std::string();            // 未知
}

// B2：在已知键中找与 raw 规范形式编辑距离最近者（简易 did-you-mean）。
// 采用 Damerau-Levenshtein（最优字符串对齐，含相邻换位），使 --gup 能提示 --gpu。
// 距离相同者取登记顺序靠前者（核心训练键排在前）。仅在足够接近时返回建议。
static std::string suggestArgKey(const std::string& raw) {
    const std::string a = canonicalArgKey(raw);
    const size_t n = a.size();
    std::string best;
    size_t bestD = (size_t)-1;
    for (const std::string& k : knownArgKeys()) {
        const std::string b = canonicalArgKey(k);
        const size_t mm = b.size();
        // d[i][j]：a 前 i 字符与 b 前 j 字符的最优对齐编辑距离
        std::vector<std::vector<size_t>> d(n + 1, std::vector<size_t>(mm + 1, 0));
        for (size_t i = 0; i <= n; ++i) d[i][0] = i;
        for (size_t j = 0; j <= mm; ++j) d[0][j] = j;
        for (size_t i = 1; i <= n; ++i) {
            for (size_t j = 1; j <= mm; ++j) {
                size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
                size_t v = d[i - 1][j - 1] + cost;
                if (d[i - 1][j] + 1 < v) v = d[i - 1][j] + 1;   // 删除
                if (d[i][j - 1] + 1 < v) v = d[i][j - 1] + 1;   // 插入
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    if (d[i - 2][j - 2] + 1 < v) v = d[i - 2][j - 2] + 1;  // 换位
                }
                d[i][j] = v;
            }
        }
        size_t dist = d[n][mm];
        if (dist < bestD) { bestD = dist; best = k; }   // 严格更小才更新 → 平局保留靠前者
    }
    if (bestD <= 2 || bestD <= best.size() / 3) return best;
    return std::string();
}

// ================= 参数解析工具 =================
// 从 argv[startIdx] 开始扫描识别 --key value 和 --flag
// 返回 map<string, string>（键名已按 B1 归一化为内部键名）
std::map<std::string, std::string> parseArgs(int argc, char** argv, int startIdx) {
    std::map<std::string, std::string> args;
    std::vector<std::string> unknownKeys;   // B2：归一化后仍未知的原始键名
    for (int i = startIdx; i < argc; ++i) {
        std::string key = argv[i];
        if (key.size() > 2 && key[0] == '-' && key[1] == '-') {
            std::string rawName = key.substr(2);

            // 先判断下一个是不是参数值（不以 -- 开头）；裸 flag 不吃掉后续 --xxx
            std::string value = "1";
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                bool isFlag = (next.size() > 2 && next[0] == '-' && next[1] == '-');
                if (!isFlag) { value = next; ++i; }
            }

            // B1：键名归一化（下游读取的内部键名不变，仅解析层做别名映射）
            std::string name = normalizeArgKey(rawName);
            if (name.empty()) {
                unknownKeys.push_back(rawName);
                args[rawName] = value;   // 保留原键（下游无引用，仅避免丢弃其值）
            } else {
                args[name] = value;
            }
        }
    }

    // B2：未知参数警告（本阶段只警告、不退出；阶段C 才收紧为报错）
    for (const std::string& u : unknownKeys) {
        std::string sug = suggestArgKey(u);
        std::cerr << "警告: 未知参数 --" << u << "（已忽略）。";
        if (!sug.empty()) std::cerr << " 是否想输入 --" << sug << " ？";
        std::cerr << "用 'cnn2 --help' 查看全部参数。\n";
    }
    return args;
}

// ================= 构建卷积核 =================
// 方案甲：Conv1 5×5×32（He 随机），Conv2 3×3×32×64（He 随机）。
// 手工 3×3 核已弃用（与 5×5 不匹配），统一改为 He 均匀初始化：
//   U(-sqrt(6/fan_in), sqrt(6/fan_in))
void buildKernels() {
    // 阶段1：逐层 He 均匀初始化，fan_in = inC*kH*kW（层 0 输入 1 通道，层 i>0 输入上一层 outC）。
    // 阶段4.1：指定 --seed 时用派生子种子（独立于洗牌/增强），保证 gen/bkc 初始化可复现；
    //           未指定时保留时间随机。
    const unsigned initSeed = g_seedSet
        ? mixSeed(g_seed, 0x1u)
        : (unsigned)std::time(nullptr);
    std::mt19937 rng(initSeed);

    int inC = 1;
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        const int OC = L.outC, KS = L.kH;
        const float lim = std::sqrt(6.0f / (float)(inC * KS * KS));
        std::uniform_real_distribution<float> dist(-lim, lim);

        KERNELS_L[i].clear();
        for (int oc = 0; oc < OC; ++oc) {
            std::vector<std::vector<std::vector<float>>> kk(
                inC, std::vector<std::vector<float>>(
                    KS, std::vector<float>(KS, 0.0f)));
            for (int c = 0; c < inC; ++c)
                for (int a = 0; a < KS; ++a)
                    for (int b = 0; b < KS; ++b)
                        kk[c][a][b] = dist(rng);
            KERNELS_L[i].push_back(kk);
        }
        BIAS_L[i].assign(OC, 0.0f);
        inC = OC;
    }

    // 阶段6：1×1 投影残差权重初始化（仅 res==2 层）。He 均匀（fan_in=srcC），偏置 0。
    // 非投影层清空，避免残留。Wproj 初始非恒等，故投影分支对输出有贡献。
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        if (L.res == 2 && L.resFrom >= 0 && L.resFrom < i) {
            const int OC = L.outC, SC = g_net.layer[L.resFrom].outC;
            const float lim = std::sqrt(6.0f / (float)SC);
            std::uniform_real_distribution<float> dist(-lim, lim);
            PROJ_K_L[i].clear();
            for (int oc = 0; oc < OC; ++oc) {
                std::vector<std::vector<std::vector<float>>> kk(
                    SC, std::vector<std::vector<float>>(
                        1, std::vector<float>(1, 0.0f)));
                for (int ic = 0; ic < SC; ++ic) kk[ic][0][0] = dist(rng);
                PROJ_K_L[i].push_back(kk);
            }
            PROJ_B_L[i].assign(OC, 0.0f);
        } else {
            PROJ_K_L[i].clear();
            PROJ_B_L[i].clear();
        }
    }

    std::cout << "第一层核数: " << KERNELS_L[0].size() << " (" << g_net.layer[0].kH << "x" << g_net.layer[0].kH << ")";
    if (g_net.depth >= 2) {
        std::cout << "，第二层核数: " << KERNELS_L[1].size()
            << " (" << g_net.layer[1].kH << "x" << g_net.layer[1].kH << "x"
            << (KERNELS_L[1].empty() ? 0 : (int)KERNELS_L[1][0].size()) << ")";
    }
    std::cout << "\n";

    // P1：BN 张量初始化。γ=1、β=0 使 BN 初始为恒等映射；
    // running_mean=0、running_var=1 使推理归一化在初始态不改变数值。
    // 非 BN 路径清空，避免上一次 BN 构建残留被误用（不影响非 BN 输出）。
    if (g_net.bn) {
        for (int i = 0; i < g_net.depth; ++i) {
            const int OC = g_net.layer[i].outC;
            BN_G_L[i].assign(OC, 1.0f);    BN_B_L[i].assign(OC, 0.0f);
            BN_MEAN_L[i].assign(OC, 0.0f); BN_VAR_L[i].assign(OC, 1.0f);
        }
    } else {
        for (int i = 0; i < g_net.depth; ++i) {
            BN_G_L[i].clear(); BN_B_L[i].clear();
            BN_MEAN_L[i].clear(); BN_VAR_L[i].clear();
        }
    }
}

// ================= 列目录 =================
std::vector<std::string> listTxtFiles(const std::string& dir) {
    std::vector<std::string> files;
    std::string pattern = dir + "\\*.txt";

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return files;

    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        files.push_back(dir + "\\" + fd.cFileName);
    } while (FindNextFileA(h, &fd));

    FindClose(h);
    return files;
}

// ================= 读图 =================
bool loadImage(const std::string& path, ConvMap& img) {
    std::ifstream fin(path);
    if (!fin) { std::cerr << "打开失败: " << path << "\n"; return false; }

    std::string line;
    int h = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (h >= img.H) { std::cerr << "行数太多\n"; return false; }
        if ((int)line.size() < img.W) {
            std::cerr << "第 " << h << " 行长度不足: " << line.size() << "\n";
            return false;
        }
        for (int w = 0; w < img.W; ++w) {
            char c = line[w];
            if (c != '0' && c != '1') {
                std::cerr << "非法字符: " << c << "\n";
                return false;
            }
            img.data[0][h][w] = (float)(c - '0');
        }
        ++h;
    }
    if (h != img.H) {
        std::cerr << "行数不符: 期望 " << img.H << "行，实际 " << h << "\n";
        return false;
    }
    return true;
}

// ================= 读图 + 预处理 =================
// 唯一带预处理的读图入口：先 loadImage 再 preprocessImage(g_prep)
static bool loadImagePreprocessed(const std::string& path, ConvMap& img) {
    if (!loadImage(path, img)) return false;

    static std::atomic<int> diagPrinted{ 0 };
    const int diagMax = 10; // 诊断输出限流，避免测试集刷屏

    PrepStats before;
    bool showDiag = g_prep.diag && (diagPrinted.load(std::memory_order_relaxed) < diagMax);
    if (showDiag) {
        before = computeStats(img);
        std::cout << "[diag] before " << path
            << " ink=" << before.inkRatio
            << " bbox=[" << before.minH << "," << before.maxH << "]x["
            << before.minW << "," << before.maxW << "]"
            << " c=(" << before.cx << "," << before.cy << ")"
            << " binary=" << (isBinary(img) ? 1 : 0) << "\n";
    }

    preprocessImage(img, g_prep);

    if (showDiag) {
        PrepStats after = computeStats(img);
        std::cout << "[diag] after  " << path
            << " ink=" << after.inkRatio
            << " bbox=[" << after.minH << "," << after.maxH << "]x["
            << after.minW << "," << after.maxW << "]"
            << " c=(" << after.cx << "," << after.cy << ")"
            << " binary=" << (isBinary(img) ? 1 : 0) << "\n";
        diagPrinted.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

// ================= 从文件路径取标签 =================
int getLabelFromPath(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    std::string name = (pos == std::string::npos) ? path : path.substr(pos + 1);
    if (name.empty() || name[0] < '0' || name[0] > '9') {
        std::cerr << "无法获取标签: " << name << "\n";
        return -1;
    }
    return name[0] - '0';
}

// ================= BN 推理（P1） =================
// 用 running 统计量做逐通道归一化（原地修改）：
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
// gamma/beta/mean/var 长度应为 x.C；通道数不符时直接返回（防御，不改动数据）。
static void bn_infer(ConvMap& x,
    const std::vector<float>& gamma, const std::vector<float>& beta,
    const std::vector<float>& mean, const std::vector<float>& var,
    float eps = CNN_BN_EPS)
{
    if ((int)gamma.size() != x.C) return;
    for (int c = 0; c < x.C; ++c) {
        const float inv = 1.0f / std::sqrt(var[c] + eps);
        const float g = gamma[c], b = beta[c], m = mean[c];
        for (int h = 0; h < x.H; ++h)
            for (int w = 0; w < x.W; ++w)
                x.data[c][h][w] = g * (x.data[c][h][w] - m) * inv + b;
    }
}

// ================= 前向 =================
// 布局：Conv(stride/pad 取自层规格) → [BN(用 running 统计量)] → ReLU → [Pool(该层 pool=2 时)]，逐层。
// g_net.bn=false 时不调用 bn_infer，逐位保持旧路径不变。
// 阶段3：层若带 identity 残差（L.res==1），在该层最终输出上逐元素加上 layer[resFrom] 的输出。
// 非残差网络保存 pLayer 与残差分支均不改变任何数值：结果与改造前逐位一致。
static void addInPlace(ConvMap& dst, const ConvMap& src) {
    if (dst.C != src.C || dst.H != src.H || dst.W != src.W) return;  // 形状由 validateNetSpec 保证
    for (int c = 0; c < dst.C; ++c)
        for (int h = 0; h < dst.H; ++h)
            for (int w = 0; w < dst.W; ++w)
                dst.data[c][h][w] += src.data[c][h][w];
}

ConvMap forwardConv(ConvMap& input) {
    // 逐层：Conv → [BN] → [残差(resPos=1 时在此加)] → ReLU → [Pool] → [残差(resPos=0 时在此加)]，
    // 保存各层输出供残差 skip 使用。
    // depth=2 默认 (stride,pad,pool)=(1,0,2) 且无残差时与改造前逐层展开完全一致。
    std::vector<ConvMap> pLayer;
    pLayer.reserve(g_net.depth);
    ConvMap cur = input;
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        ConvMap c = conv(cur, KERNELS_L[i], BIAS_L[i], L.stride, L.pad);
        if (g_net.bn) bn_infer(c, BN_G_L[i], BN_B_L[i], BN_MEAN_L[i], BN_VAR_L[i]);
        // h 为 ReLU 输入（= BN 输出，或 pre-activation 残差叠加后的值）
        ConvMap h = c;
        const bool haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < (int)pLayer.size());
        if (haveSrc && L.resPos == 1) {
            if (L.res == 1) addInPlace(h, pLayer[L.resFrom]);
            else {
                ConvMap proj = conv(pLayer[L.resFrom], PROJ_K_L[i], PROJ_B_L[i], 1, 0);
                addInPlace(h, proj);
            }
        }
        ConvMap r = relu(h);
        ConvMap p = L.pool ? Pool2x2(r) : r;
        if (haveSrc && L.resPos == 0) {
            if (L.res == 1) addInPlace(p, pLayer[L.resFrom]);
            else {
                ConvMap proj = conv(pLayer[L.resFrom], PROJ_K_L[i], PROJ_B_L[i], 1, 0);
                addInPlace(p, proj);
            }
        }
        pLayer.push_back(std::move(p));
        cur = pLayer.back();
    }
    return cur;
}

// ================= 保存 / 加载 模型 =================
// T3：生成与 txt 首行语义一致的 #SIG 串（保存 acm 头与 txt 首行复用同一逻辑）。
static std::string currentModelSig() {
    // 头部强制写（P0 起不再依赖 --P）：始终输出 #SIG 行。
    //   续训：保留旧设备/时间字段，强制刷新 BN:/depth=/ver；
    //   新建：--P 写完整环境，否则仅 BN+depth。
    // 阶段2：depth>2 或存在非默认 P<i> 时追加 "ver=2"。
    bool needVer2 = (g_net.depth > 2);
    for (int i = 0; i < g_net.depth; ++i)
        if (!isDefaultP(g_net.layer[i]) || g_net.layer[i].res != 0 ||
            g_net.layer[i].resPos != 0) { needVer2 = true; break; }
    const int outVer = needVer2 ? 2 : 1;
    if (!g_loadedSig.empty())
        return normalizeSignature(g_loadedSig, g_net.bn, g_net.depth, outVer);
    return makeSignature(g_net.bn, g_net.depth, g_modelSign, outVer);
}

// 原文本实现（逻辑未变）；统一入口见文件后部 saveModel()。
bool saveModelTxt(const std::string& path, const FullyConnected& fc) {
    std::ofstream fout(path);
    if (!fout) { std::cerr << "打不开写入: " << path << "\n"; return false; }

    // 头部强制写（P0 起不再依赖 --P）：始终输出 #SIG 行。
    fout << currentModelSig() << "\n";

    // 逐层写 K<i>/B<i>/[BN<i>]/[P<i>]（1-based tag，depth=2 默认时与改造前逐字节一致）
    for (int i = 0; i < g_net.depth; ++i) {
        const K4& K = KERNELS_L[i];
        const std::vector<float>& B = BIAS_L[i];
        int o  = (int)K.size();
        int ic = (o > 0) ? (int)K[0].size() : 0;
        int h  = (ic > 0 && !K[0].empty()) ? (int)K[0][0].size() : 0;
        int w  = (h > 0 && !K[0][0].empty()) ? (int)K[0][0][0].size() : 0;
        fout << "K" << (i + 1) << " " << o << " " << ic << " " << h << " " << w << "\n";
        for (auto& kk : K)
            for (auto& ch : kk)
                for (auto& row : ch)
                    for (float v : row) fout << v << " ";
        fout << "\n";

        fout << "B" << (i + 1) << " " << B.size() << "\n";
        for (float v : B) fout << v << " ";
        fout << "\n";

        // BN<i> 行（仅 BN 模型）：BN<i> <outC>，随后依次 outC 个 γ、β、mean、var。
        if (g_net.bn) {
            fout << "BN" << (i + 1) << " " << g_net.layer[i].outC << "\n";
            for (float v : BN_G_L[i])    fout << v << " "; fout << "\n";
            for (float v : BN_B_L[i])    fout << v << " "; fout << "\n";
            for (float v : BN_MEAN_L[i]) fout << v << " "; fout << "\n";
            for (float v : BN_VAR_L[i])  fout << v << " "; fout << "\n";
        }

        // P<i> 行（阶段2）：depth>2 时每层都写；depth=2 时仅非默认层写。
        if (g_net.depth > 2 || !isDefaultP(g_net.layer[i])) {
            const LayerSpec& L = g_net.layer[i];
            fout << "P" << (i + 1) << " " << L.stride << " " << L.pad << " " << L.pool << "\n";
        }

        // R<i> 行（阶段3/6，仅残差层写）：R<i> <res> <resFrom> [<resPos>]。
        // resPos 仅在非默认（!=0）时追加，旧文件/旧语言保持三字段，逐字节兼容。
        const LayerSpec& RL = g_net.layer[i];
        if (RL.res != 0) {
            fout << "R" << (i + 1) << " " << RL.res << " " << RL.resFrom;
            if (RL.resPos != 0) fout << " " << RL.resPos;
            fout << "\n";
        }

        // RP<i>/PB<i>（阶段6，仅 res==2 层写）：1×1 投影权重（outC×srcC）+ 偏置（outC）。
        // 用新 tag RP/PB；旧程序读到 R<i> 的 res=2 即干净报错，不会误读这两段。
        if (RL.res == 2 && RL.resFrom >= 0 && RL.resFrom < i) {
            const int OC = g_net.layer[i].outC;
            const int SC = g_net.layer[RL.resFrom].outC;
            fout << "RP" << (i + 1) << " " << OC << " " << SC << "\n";
            for (int oc = 0; oc < OC; ++oc)
                for (int ic = 0; ic < SC; ++ic)
                    fout << PROJ_K_L[i][oc][ic][0][0] << " ";
            fout << "\n";
            fout << "PB" << (i + 1) << " " << OC << "\n";
            for (int oc = 0; oc < OC; ++oc) fout << PROJ_B_L[i][oc] << " ";
            fout << "\n";
        }
    }

    fout << "FC " << fc.outDim << " " << fc.inDim << "\n";
    for (int o = 0; o < fc.outDim; ++o) {
        for (int i = 0; i < fc.inDim; ++i)
            fout << fc.W[o][i] << " ";
        fout << "\n";
    }
    for (int o = 0; o < fc.outDim; ++o)
        fout << fc.b[o] << " ";
    fout << "\n";

    return true;
}

// 原文本实现（逻辑未变）；统一入口见文件后部 loadModel()。
bool loadModelTxt(const std::string& path, FullyConnected& fc) {
    std::ifstream fin(path);
    if (!fin) { std::cerr << "打不开读取: " << path << "\n"; return false; }

    std::string tag;

    // M2：解析与内存分配整段包在 try/catch 内，坏维度/超大分配不再 terminate
    try {

    // 首行检测：以 '#' 开头视为签名行并跳过；否则回退文件开头，兼容旧格式
    {
        std::streampos pos0 = fin.tellg();
        std::string first;
        std::getline(fin, first);
        // 剥离 UTF-8 BOM（防御性）
        if (first.size() >= 3 &&
            (unsigned char)first[0] == 0xEF &&
            (unsigned char)first[1] == 0xBB &&
            (unsigned char)first[2] == 0xBF) first.erase(0, 3);
        if (!first.empty() && first.back() == '\r') first.pop_back();
        size_t b = first.find_first_not_of(" \t");
        if (b != std::string::npos && first[b] == '#') {
            g_loadedSig = first.substr(b);   // 记录原签名（续训保留用）
            // P0：解析 #SIG 中的 BN/depth；旧签名无 BN: 字段则归一化为 BN:OFF。
            // peek 已按同一签名设好 g_net，这里再同步一次以便读取 BN 数据行。
            bool sigBn = false; int sigDepth = 2;
            parseSigFields(g_loadedSig, sigBn, sigDepth);
            g_net.bn = sigBn;
            // 注意：depth 以实际读到的层数为准（见循环后），此处仅暂存签名值，避免降级。
            g_net.depth = sigDepth;
        } else {
            fin.clear();
            fin.seekg(pos0);                 // 旧格式：回到开头
        }
    }

    fin >> tag;
    // 阶段1：逐层读 K<i>/B<i>/[BN<i>]，直到遇到 FC。
    // tag 前缀匹配、索引连续（K1,B1,K2,B2,...）；i_{n+1}==o_n，预读的 tag 不丢。
    int layerCount = 0;
    int prevOut = 1;
    while (tag != "FC") {
        if (layerCount >= MAX_DEPTH) { std::cerr << "层数超过 MAX_DEPTH(" << MAX_DEPTH << ")\n"; return false; }
        const std::string wantK = "K" + std::to_string(layerCount + 1);
        if (tag != wantK) { std::cerr << "缺少 " << wantK << "（读到 " << tag << "）\n"; return false; }
        int o, ic, h, w;
        fin >> o >> ic >> h >> w;
        // M2：分配前校验维度为正，负数不会进入 assign（否则 size_t 巨值 → length_error/terminate）
        if (o <= 0 || ic <= 0 || h <= 0 || w <= 0) {
            std::cerr << wantK << " 维度非法: " << o << "x" << ic << "x" << h << "x" << w << "\n";
            return false;
        }
        if (layerCount > 0 && ic != prevOut) {
            std::cerr << wantK << " 输入通道与上一层输出通道不一致\n"; return false;
        }
        KERNELS_L[layerCount].assign(o,
            std::vector<std::vector<std::vector<float>>>(
                ic, std::vector<std::vector<float>>(
                    h, std::vector<float>(w, 0.0f))));
        for (auto& kk : KERNELS_L[layerCount])
            for (auto& ch : kk)
                for (auto& row : ch)
                    for (float& v : row) fin >> v;

        const std::string wantB = "B" + std::to_string(layerCount + 1);
        fin >> tag;
        if (tag != wantB) { std::cerr << "缺少 " << wantB << "\n"; return false; }
        int nb;
        fin >> nb;
        if (nb <= 0) { std::cerr << wantB << " 维度非法: " << nb << "\n"; return false; }
        BIAS_L[layerCount].assign(nb, 0.0f);
        for (float& v : BIAS_L[layerCount]) fin >> v;

        // 可选 BN<i>：仅当实际读到才读取（旧模型无此行自然回退到下一 tag）。
        fin >> tag;
        const std::string wantBN = "BN" + std::to_string(layerCount + 1);
        if (tag == wantBN) {
            int nbn = 0; fin >> nbn;
            if (nbn <= 0) { std::cerr << wantBN << " 维度非法: " << nbn << "\n"; return false; }
            BN_G_L[layerCount].assign(nbn, 1.0f);    BN_B_L[layerCount].assign(nbn, 0.0f);
            BN_MEAN_L[layerCount].assign(nbn, 0.0f); BN_VAR_L[layerCount].assign(nbn, 1.0f);
            for (float& v : BN_G_L[layerCount])    fin >> v;
            for (float& v : BN_B_L[layerCount])    fin >> v;
            for (float& v : BN_MEAN_L[layerCount]) fin >> v;
            for (float& v : BN_VAR_L[layerCount])  fin >> v;
            fin >> tag;
        }

        // 可选 P<i>：阶段2。缺失则沿用默认 (1,0,2)（由 peek 设定的层规格）。
        const std::string wantP = "P" + std::to_string(layerCount + 1);
        if (tag == wantP) {
            int st = 1, pd = 0, pl = 2;
            fin >> st >> pd >> pl;
            if (st < 1 || pd < 0 || (pl != 0 && pl != 2)) {
                std::cerr << wantP << " 参数非法（stride>=1, pad>=0, pool∈{0,2}）: "
                          << st << " " << pd << " " << pl << "\n";
                return false;
            }
            g_net.layer[layerCount].stride = st;
            g_net.layer[layerCount].pad = pd;
            g_net.layer[layerCount].pool = pl;
            fin >> tag;
        }

        // 可选 R<i>（阶段3/6，仅残差层有）：R<i> <res> <resFrom> [<resPos>]；缺失则无残差。
        // resPos 仅为新能力，缺省=0；通过「下一字符是否为数字」探测，旧三字段文件不受影响。
        const std::string wantR = "R" + std::to_string(layerCount + 1);
        if (tag == wantR) {
            int rs = 0, rf = -1, rp = 0;
            fin >> rs >> rf;
            if (rs != 0 && rs != 1 && rs != 2) {
                std::cerr << wantR << " 残差类型非法（仅支持 0/1/2）: " << rs << "\n";
                return false;
            }
            fin >> std::ws;
            {
                const int pk = fin.peek();
                if (pk == '-' || (pk >= '0' && pk <= '9')) { fin >> rp; }
            }
            if (rp != 0 && rp != 1) {
                std::cerr << wantR << " 残差位置非法（仅支持 0/1）: " << rp << "\n";
                return false;
            }
            g_net.layer[layerCount].res = rs;
            g_net.layer[layerCount].resFrom = rf;
            g_net.layer[layerCount].resPos = rp;
            fin >> tag;

            // res==2：紧随 R<i> 之后读 RP<i>（1×1 权重）与 PB<i>（偏置）。
            if (rs == 2) {
                if (rf < 0 || rf >= layerCount) {
                    std::cerr << wantR << " 投影残差源层索引非法: " << rf << "\n";
                    return false;
                }
                const std::string wantRP = "RP" + std::to_string(layerCount + 1);
                if (tag != wantRP) {
                    std::cerr << "缺少 " << wantRP << "（读到 " << tag << "）\n";
                    return false;
                }
                int po = 0, pc = 0;
                fin >> po >> pc;
                const int expO = g_net.layer[layerCount].outC;
                const int expC = g_net.layer[rf].outC;
                if (po != expO || pc != expC) {
                    std::cerr << wantRP << " 维度不符: " << po << "x" << pc
                              << "（期望 " << expO << "x" << expC << "）\n";
                    return false;
                }
                // 用「文件中的 outC/inC」重建层内通道关系，供后续 validate 一致性
                PROJ_K_L[layerCount].assign(po,
                    std::vector<std::vector<std::vector<float>>>(
                        pc, std::vector<std::vector<float>>(
                            1, std::vector<float>(1, 0.0f))));
                for (int oc = 0; oc < po; ++oc)
                    for (int ic = 0; ic < pc; ++ic)
                        fin >> PROJ_K_L[layerCount][oc][ic][0][0];

                const std::string wantPB = "PB" + std::to_string(layerCount + 1);
                fin >> tag;
                if (tag != wantPB) {
                    std::cerr << "缺少 " << wantPB << "（读到 " << tag << "）\n";
                    return false;
                }
                int pb = 0; fin >> pb;
                if (pb != expO) {
                    std::cerr << wantPB << " 维度不符: " << pb << "（期望 " << expO << "）\n";
                    return false;
                }
                PROJ_B_L[layerCount].assign(pb, 0.0f);
                for (float& v : PROJ_B_L[layerCount]) fin >> v;
                fin >> tag;
            }
        }
        // 非投影层清空投影缓冲，避免多次加载时残留被误用。
        if (g_net.layer[layerCount].res != 2) {
            PROJ_K_L[layerCount].clear();
            PROJ_B_L[layerCount].clear();
        }
        prevOut = o;
        ++layerCount;
    }

    // 防坑：depth 以实际读到的层数为准，绝不被 #SIG 的旧值静默降级/升级。
    g_net.depth = layerCount;

    int outD, inD;
    fin >> outD >> inD;
    // 阶段0+1：不再强校验预先为 2304 造好的 fc，而是按文件解析出的维度重新分配，
    // 从而支持任意规模权重文件（调用方应已用 peekModelDims 设好 g_net）。
    if (outD <= 0 || inD <= 0) {
        std::cerr << "FC 维度非法: " << outD << "x" << inD << "\n";
        return false;
    }
    fc.outDim = outD;
    fc.inDim = inD;
    fc.W.assign(outD, std::vector<float>(inD, 0.0f));
    fc.b.assign(outD, 0.0f);
    for (int o = 0; o < outD; ++o)
        for (int i = 0; i < inD; ++i)
            fin >> fc.W[o][i];
    for (int o = 0; o < outD; ++o)
        fin >> fc.b[o];

    return true;
    } catch (const std::bad_alloc&) {
        std::cerr << "加载模型失败: 内存分配不足 (" << path << ")\n";
        return false;
    } catch (const std::length_error& e) {
        std::cerr << "加载模型失败: 维度超出可分配范围 (" << path << "): " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "加载模型失败: " << path << " : " << e.what() << "\n";
        return false;
    }
}

// ================= GPU training: flatten / scatter params =================
// 扁平参数布局（与 gpu_kernels.cuh 的 OFF_*/N_* 严格一致，方案甲）：
//   [0,800)       K1   (CONV1: 32×5×5)
//   [800,832)     B1   (32)
//   [832,19264)   K2   (CONV2: 64×32×3×3)
//   [19264,19328) B2   (64)
//   [19328,42368) FC_W (10×2304, row-major)
//   [42368,42378) FC_b (10)
// 阶段0+1：以上布局的偏移/数量不再硬编码，全部来自运行时规格 g_net。
static void flattenParams(std::vector<float>& p, const FullyConnected& fc) {
    p.assign(g_net.total, 0.0f);

    // 逐层：K<i> | B<i> | [BN<i> g/b/mean/var]；写序与 makeDefaultNetSpec 的偏移一致。
    for (int i = 0; i < g_net.depth; ++i) {
        const K4& K = KERNELS_L[i];
        int k = 0;
        for (int oc = 0; oc < (int)K.size(); ++oc)
            for (int ic = 0; ic < (int)K[oc].size(); ++ic)
                for (int kh = 0; kh < (int)K[oc][ic].size(); ++kh)
                    for (int kw = 0; kw < (int)K[oc][ic][kh].size(); ++kw)
                        p[g_net.offK[i] + k++] = K[oc][ic][kh][kw];

        for (int j = 0; j < (int)BIAS_L[i].size() && j < g_net.nB[i]; ++j)
            p[g_net.offB[i] + j] = BIAS_L[i][j];

        // BN<i> 段（γ/β/mean/var）。非 BN 时各数量为 0，循环不执行，行为不变。
        for (int j = 0; j < g_net.nBNg[i];    ++j) p[g_net.offBNg[i]    + j] = BN_G_L[i][j];
        for (int j = 0; j < g_net.nBNb[i];    ++j) p[g_net.offBNb[i]    + j] = BN_B_L[i][j];
        for (int j = 0; j < g_net.nBNmean[i]; ++j) p[g_net.offBNmean[i] + j] = BN_MEAN_L[i][j];
        for (int j = 0; j < g_net.nBNvar[i];  ++j) p[g_net.offBNvar[i]  + j] = BN_VAR_L[i][j];

        // 阶段6：投影段（仅 res==2；非投影层数量为 0，不写）。
        {
            int k = 0;
            for (int oc = 0; oc < (int)PROJ_K_L[i].size(); ++oc)
                for (int ic = 0; ic < (int)PROJ_K_L[i][oc].size(); ++ic)
                    for (int kh = 0; kh < (int)PROJ_K_L[i][oc][ic].size(); ++kh)
                        for (int kw = 0; kw < (int)PROJ_K_L[i][oc][ic][kh].size(); ++kw)
                            if (k < g_net.nPW[i]) p[g_net.offPW[i] + k++] = PROJ_K_L[i][oc][ic][kh][kw];
            for (int j = 0; j < (int)PROJ_B_L[i].size() && j < g_net.nPB[i]; ++j)
                p[g_net.offPB[i] + j] = PROJ_B_L[i][j];
        }
    }

    for (int o = 0; o < fc.outDim; ++o)
        for (int i = 0; i < fc.inDim; ++i)
            p[g_net.offFCW + o * g_net.flat + i] = fc.W[o][i];

    for (int o = 0; o < fc.outDim; ++o) p[g_net.offFCB + o] = fc.b[o];
}

static void scatterParams(const float* p, FullyConnected& fc) {
    // 逐层：K<i> | B<i> | [BN<i> g/b/mean/var]；读序与 makeDefaultNetSpec 的偏移一致。
    for (int i = 0; i < g_net.depth; ++i) {
        K4& K = KERNELS_L[i];
        int k = 0;
        for (int oc = 0; oc < (int)K.size(); ++oc)
            for (int ic = 0; ic < (int)K[oc].size(); ++ic)
                for (int kh = 0; kh < (int)K[oc][ic].size(); ++kh)
                    for (int kw = 0; kw < (int)K[oc][ic][kh].size(); ++kw)
                        K[oc][ic][kh][kw] = p[g_net.offK[i] + k++];

        for (int j = 0; j < (int)BIAS_L[i].size() && j < g_net.nB[i]; ++j)
            BIAS_L[i][j] = p[g_net.offB[i] + j];

        // BN<i> 段拷出（非 BN 时数量为 0，循环不执行，行为不变）。
        for (int j = 0; j < g_net.nBNg[i];    ++j) BN_G_L[i][j]    = p[g_net.offBNg[i]    + j];
        for (int j = 0; j < g_net.nBNb[i];    ++j) BN_B_L[i][j]    = p[g_net.offBNb[i]    + j];
        for (int j = 0; j < g_net.nBNmean[i]; ++j) BN_MEAN_L[i][j] = p[g_net.offBNmean[i] + j];
        for (int j = 0; j < g_net.nBNvar[i];  ++j) BN_VAR_L[i][j]  = p[g_net.offBNvar[i]  + j];

        // 阶段6：投影段拷出（仅 res==2；非投影层数量为 0，不读）。
        {
            int k = 0;
            for (int oc = 0; oc < (int)PROJ_K_L[i].size(); ++oc)
                for (int ic = 0; ic < (int)PROJ_K_L[i][oc].size(); ++ic)
                    for (int kh = 0; kh < (int)PROJ_K_L[i][oc][ic].size(); ++kh)
                        for (int kw = 0; kw < (int)PROJ_K_L[i][oc][ic][kh].size(); ++kw)
                            if (k < g_net.nPW[i]) PROJ_K_L[i][oc][ic][kh][kw] = p[g_net.offPW[i] + k++];
            for (int j = 0; j < (int)PROJ_B_L[i].size() && j < g_net.nPB[i]; ++j)
                PROJ_B_L[i][j] = p[g_net.offPB[i] + j];
        }
    }

    for (int o = 0; o < fc.outDim; ++o)
        for (int i = 0; i < fc.inDim; ++i)
            fc.W[o][i] = p[g_net.offFCW + o * g_net.flat + i];

    for (int o = 0; o < fc.outDim; ++o) fc.b[o] = p[g_net.offFCB + o];
}

// ================= 权重文件维度预览（阶段0+1） =================
// 只解析权重文件中 K1/K2/FC 三行的形状（跳过 K1/B1/K2/B2 的数值），
// 构造 NetSpec 并校验，不读取全部权重、不修改全局权重。
// 首行跳过 #SIG 签名的逻辑与 loadModel 完全一致（loadModel 的签名逻辑保持不动）。
// 原文本实现（逻辑未变）；统一入口见文件后部 peekModelDims()。
static bool peekModelDimsTxt(const std::string& path, NetSpec& out,
                          std::string* err = nullptr) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    std::ifstream fin(path);
    if (!fin) return fail("打不开文件");

    // P0：从 #SIG 解析 BN/depth（旧签名缺省 BN:OFF / depth=2）
    bool sigBn = false; int sigDepth = 2;

    // ---- 与 loadModel 相同：首行以 '#' 开头视为签名行并跳过，否则回退文件开头 ----
    {
        std::streampos pos0 = fin.tellg();
        std::string first;
        std::getline(fin, first);
        if (first.size() >= 3 &&
            (unsigned char)first[0] == 0xEF &&
            (unsigned char)first[1] == 0xBB &&
            (unsigned char)first[2] == 0xBF) first.erase(0, 3);
        if (!first.empty() && first.back() == '\r') first.pop_back();
        size_t b = first.find_first_not_of(" \t");
        if (b != std::string::npos && first[b] == '#') {
            parseSigFields(first.substr(b), sigBn, sigDepth);
        } else {
            fin.clear();
            fin.seekg(pos0);
        }
    }

    auto skipFloats = [&](long long n) -> bool {
        for (long long i = 0; i < n; ++i) { float v; if (!(fin >> v)) return false; }
        return true;
    };

    std::string tag;
    // 阶段1：逐层读 K<i>/B<i>/[BN<i>] 直到 FC（只跳过数值，不读全部权重）。
    // tag 前缀匹配；索引连续（K1,B1,K2,B2,...）；预读的 tag 不丢；i_{n+1}==o_n。
    int lc = 0;
    int prevOut = 1;
    int layerO[MAX_DEPTH] = {}, layerK[MAX_DEPTH] = {};
    int layerStride[MAX_DEPTH] = {}, layerPad[MAX_DEPTH] = {}, layerPool[MAX_DEPTH] = {};
    int layerRes[MAX_DEPTH] = {}, layerResFrom[MAX_DEPTH] = {};
    int layerResPos[MAX_DEPTH] = {};
    for (int i = 0; i < MAX_DEPTH; ++i) { layerStride[i] = 1; layerPad[i] = 0; layerPool[i] = 2; layerResFrom[i] = -1; }
    if (!(fin >> tag)) return fail("缺少 K1");
    while (tag != "FC") {
        if (lc >= MAX_DEPTH) return fail("层数超过 MAX_DEPTH");
        const std::string wantK = "K" + std::to_string(lc + 1);
        if (tag != wantK) return fail("缺少 K 段");
        int o, ic, h, w;
        if (!(fin >> o >> ic >> h >> w)) return fail("K 维度不完整");
        if (o <= 0 || ic <= 0 || h <= 0 || w <= 0) return fail("K 维度必须为正");
        if (lc == 0) { if (ic != 1) return fail("K1 输入通道必须为 1"); }
        else if (ic != prevOut) return fail("层间输入/输出通道不一致");
        if (h != w) return fail("核必须为方形");
        if (!skipFloats((long long)o * ic * h * w)) return fail("K 数据不完整");
        layerO[lc] = o; layerK[lc] = h; prevOut = o; ++lc;

        const std::string wantB = "B" + std::to_string(lc);
        if (!(fin >> tag) || tag != wantB) return fail("缺少 B 段");
        int nb = 0;
        if (!(fin >> nb)) return fail("B 维度不完整");
        if (nb <= 0) return fail("B 维度必须为正");
        if (!skipFloats(nb)) return fail("B 数据不完整");

        // 可选 BN<i>：仅当实际读到才跳过（旧模型无此行自然回退）
        if (!(fin >> tag)) return fail("缺少 FC");
        const std::string wantBN = "BN" + std::to_string(lc);
        if (tag == wantBN) {
            int nbn = 0;
            if (!(fin >> nbn)) return fail("BN 维度不完整");
            if (nbn <= 0) return fail("BN 维度必须为正");
            if (!skipFloats((long long)nbn * 4)) return fail("BN 数据不完整");
            if (!(fin >> tag)) return fail("缺少 FC");
        }

        // 可选 P<i> <stride> <pad> <pool>：缺失则默认 (1,0,2)（阶段2）。
        const std::string wantP = "P" + std::to_string(lc);
        if (tag == wantP) {
            int st = 1, pd = 0, pl = 2;
            if (!(fin >> st >> pd >> pl)) return fail("P 参数不完整");
            if (st < 1 || pd < 0 || (pl != 0 && pl != 2)) return fail("P 参数非法（stride>=1,pad>=0,pool∈{0,2}）");
            layerStride[lc - 1] = st; layerPad[lc - 1] = pd; layerPool[lc - 1] = pl;
            if (!(fin >> tag)) return fail("缺少 FC");
        }

        // 可选 R<i>（阶段3/6）：R<i> <res> <resFrom> [<resPos>]；缺失则无残差。旧文件自然回退。
        // res=2 时其后紧跟 RP<i>（outC×srcC 值）与 PB<i>（outC 值），在此一并跳过。
        const std::string wantR = "R" + std::to_string(lc);
        if (tag == wantR) {
            int rs = 0, rf = -1, rp = 0;
            if (!(fin >> rs >> rf)) return fail("R 参数不完整");
            if (rs != 0 && rs != 1 && rs != 2) return fail("R 残差类型非法（仅支持 0/1/2）");
            fin >> std::ws;
            {
                const int pk = fin.peek();
                if (pk == '-' || (pk >= '0' && pk <= '9')) { if (!(fin >> rp)) return fail("R 位置不完整"); }
            }
            if (rp != 0 && rp != 1) return fail("R 残差位置非法（仅支持 0/1）");
            layerRes[lc - 1] = rs; layerResFrom[lc - 1] = rf; layerResPos[lc - 1] = rp;
            if (!(fin >> tag)) return fail("缺少 FC");
            if (rs == 2) {
                const std::string wantRP = "RP" + std::to_string(lc);
                if (tag != wantRP) return fail("缺少 RP 投影权重段");
                int po = 0, pc = 0;
                if (!(fin >> po >> pc)) return fail("RP 维度不完整");
                if (po <= 0 || pc <= 0) return fail("RP 维度必须为正");
                if (!skipFloats((long long)po * pc)) return fail("RP 数据不完整");
                const std::string wantPB = "PB" + std::to_string(lc);
                if (!(fin >> tag) || tag != wantPB) return fail("缺少 PB 投影偏置段");
                int pb = 0;
                if (!(fin >> pb)) return fail("PB 维度不完整");
                if (pb <= 0) return fail("PB 维度必须为正");
                if (!skipFloats(pb)) return fail("PB 数据不完整");
                if (!(fin >> tag)) return fail("缺少 FC");
            }
        }
    }

    int outD, inD;
    if (!(fin >> outD >> inD)) return fail("FC 维度不完整");
    if (outD <= 0 || inD <= 0) return fail("FC 维度必须为正");

    // 阶段2：以实际读到的层数 lc 作为 depth（[1, MAX_DEPTH]）。#SIG 的 sigDepth 仅作参考。
    // 输入固定 32×32（权重文件不含输入尺寸）；由文件形状 + P<i> 构造完整规格后整体校验。
    NetSpec s;
    s.inH = 32; s.inW = 32; s.classes = outD;
    s.bn = sigBn; s.depth = lc;
    for (int i = 0; i < lc && i < MAX_DEPTH; ++i) {
        LayerSpec& L = s.layer[i];
        L.outC = layerO[i];
        L.inC = (i == 0) ? 1 : layerO[i - 1];
        L.kH = L.kW = layerK[i];
        L.stride = layerStride[i]; L.pad = layerPad[i]; L.pool = layerPool[i];
        L.bn = sigBn ? 1 : 0;
        L.res = layerRes[i]; L.resFrom = layerResFrom[i]; L.resPos = layerResPos[i];
    }
    computeNetSpec(s);
    if (!validateNetSpec(s, err)) return false;
    if (inD != s.flat) return fail("FC 输入维度与卷积展平不一致");
    out = s;
    return true;
}

// ================= T3/T4：模型格式分流（txt 文本 / acm 二进制） =================
// 命名约定：*Txt 为原有文本实现（逻辑未变）；*Acm 为新增二进制实现；
//           无后缀的统一入口 saveModel/loadModel/peekModelDims 负责分派：
//           - 写侧：按文件名后缀 + g_saveAcm（T5 规则，见下）；
//           - 读侧：恒用 probeModelFormat（优先魔数）自动识别，不受开关影响。

// T3：acm 二进制保存。布局 = 1024B 固定头 + total×float32 数据区（NetSpec 扁平序）。
bool saveModelAcm(const std::string& path, const FullyConnected& fc) {
    const std::string sig = currentModelSig();

    std::vector<float> params;
    flattenParams(params, fc);                 // 大小 = g_net.total（NetSpec 扁平布局）

    std::vector<uint8_t> hdr;
    std::string err;
    if (!encodeAcmHeader(g_net, sig, /*precision=*/0, hdr, &err)) {
        std::cerr << "acm 保存失败: " << err << "\n";
        return false;
    }

    // 数据区（小端 float32；Windows/x64 为小端，直接按字节落盘）
    std::vector<uint8_t> data((size_t)g_net.total * sizeof(float));
    if (!params.empty())
        std::memcpy(data.data(), params.data(), data.size());

    // 数据区 CRC32 -> 头部偏移 40，并置 flags 的 CRC32 标志位
    const uint32_t crc = acmCrc32(data.data(), data.size());
    acmPutU32(hdr.data() + 40, crc);
    acmPutU16(hdr.data() + 8, (uint16_t)(acmGetU16(hdr.data() + 8) | ACM_FLAG_CRC32));

    std::ofstream fout(path, std::ios::binary);
    if (!fout) { std::cerr << "打不开写入: " << path << "\n"; return false; }
    fout.write(reinterpret_cast<const char*>(hdr.data()), (std::streamsize)hdr.size());
    fout.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());
    if (!fout) { std::cerr << "写模型失败: " << path << "\n"; return false; }
    return true;
}

// ================= T7：生成路径后缀随格式 =================
// 程序「生成」的路径（model_epochN / ep_<epoch> / <model>.best / 默认模型名）后缀随格式：
// 依据 T5 规则（.txt->文本；.acm->二进制；无/未知后缀->按 g_saveAcm，默认 acm）。
static ModelFormat currentSaveFormat() {
    return g_saveAcm ? ModelFormat::Acm : ModelFormat::Txt;
}
// 依据给定路径后缀 + g_saveAcm 判定「保存格式」（与 saveModel 分流规则保持一致）。
static ModelFormat modelSaveFormat(const std::string& path) {
    const std::string ext = pathExtLower(path);
    if (ext == ".txt") return ModelFormat::Txt;
    if (ext == ".acm") return ModelFormat::Acm;
    return currentSaveFormat();
}

// T3 统一入口（T5 后缀规则）：
//   .txt -> 文本；.acm -> 二进制；.acm + --txtmodel/环境变量 -> 冲突报错；
//   无/其它后缀 -> 按 g_saveAcm（true=acm，false=txt）。
bool saveModel(const std::string& path, const FullyConnected& fc) {
    const std::string ext = pathExtLower(path);
    bool useAcm;
    if (ext == ".txt") {
        useAcm = false;                        // 后缀决定：文本
    } else if (ext == ".acm") {
        if (!g_saveAcm) {
            std::cerr << "错误: 后缀 .acm 与 --txtmodel/环境变量 CNN2_TXTMODEL 冲突"
                         "（.acm 表示二进制格式）: " << path << "\n";
            return false;
        }
        useAcm = true;                         // 后缀决定：二进制
    } else {
        useAcm = g_saveAcm;                    // 无/未知后缀：按开关（默认 acm）
    }
    return useAcm ? saveModelAcm(path, fc) : saveModelTxt(path, fc);
}

// T4：acm 二进制加载（头 + 数据区 -> 全局权重 + fc）。精度非 fp32 由 decode 报错退出。
bool loadModelAcm(const std::string& path, FullyConnected& fc) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { std::cerr << "打不开读取: " << path << "\n"; return false; }

    fin.seekg(0, std::ios::end);
    const std::streamoff sz = fin.tellg();
    if (sz < (std::streamoff)ACM_HEADER_SIZE) {
        std::cerr << "加载模型失败: acm 头部不完整 (" << path << ")\n";
        return false;
    }
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t)sz, 0);
    fin.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)sz);

    AcmHeader h; NetSpec spec; std::string sig; std::string err;
    if (!decodeAcmHeader(buf, h, spec, sig, &err)) {
        std::cerr << "加载模型失败: " << err << " (" << path << ")\n";
        return false;
    }
    const size_t need = (size_t)ACM_HEADER_SIZE + (size_t)spec.total * sizeof(float);
    if (buf.size() < need) {
        std::cerr << "加载模型失败: acm 数据区不完整（期望 " << need
                  << " 字节，实际 " << buf.size() << "）: " << path << "\n";
        return false;
    }

    // R2/T3：若头部置位 CRC32 标志（bit0），重算数据区 CRC 与头部偏移 40 比对，
    //        不符则判定文件损坏并拒绝加载（写侧 saveModelAcm 恒写该字段与标志位）。
    if (h.flags & ACM_FLAG_CRC32) {
        const uint32_t actual =
            acmCrc32(buf.data() + ACM_HEADER_SIZE, (size_t)spec.total * sizeof(float));
        if (actual != h.crc32) {
            std::cerr << "加载模型失败: acm 数据区 CRC32 校验失败（头部=" << h.crc32
                      << "，实际=" << actual << "）: " << path << "\n";
            return false;
        }
    }

    // 同步运行时规格与签名（与 txt 续训语义一致：保留原 #SIG 串）
    g_net = spec;
    g_loadedSig = sig;

    // 按 g_net 分配 FC 维度（与 scatter 布局一致）
    fc.outDim = g_net.classes;
    fc.inDim = g_net.flat;
    fc.W.assign(fc.outDim, std::vector<float>(fc.inDim, 0.0f));
    fc.b.assign(fc.outDim, 0.0f);

    std::vector<float> params((size_t)g_net.total);
    std::memcpy(params.data(), buf.data() + ACM_HEADER_SIZE, params.size() * sizeof(float));
    scatterParams(params.data(), fc);
    return true;
}

// T4 统一入口：读侧恒按魔数自动识别（不受 --txtmodel 影响）。
// R1：显式 .acm 后缀但魔数不符（=损坏）判为 Unknown，直接报错，绝不回退按 txt 解析
//     （错误信息与 read 命令一致，避免误导性报错）。
bool loadModel(const std::string& path, FullyConnected& fc) {
    const ModelFormat fmt = probeModelFormat(path);
    if (fmt == ModelFormat::Acm) return loadModelAcm(path, fc);
    if (fmt == ModelFormat::Unknown) {
        std::cerr << "加载模型失败: " << path
                  << " 后缀为 .acm 但魔数不符（文件损坏或并非 acm 格式）\n";
        return false;
    }
    return loadModelTxt(path, fc);
}

// T4：acm 维度预览（只读 1024B 头，不读数据区）。
static bool peekModelDimsAcm(const std::string& path, NetSpec& out,
                             std::string* err = nullptr) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return fail("打不开文件");
    std::vector<uint8_t> hdr((size_t)ACM_HEADER_SIZE, 0);
    fin.read(reinterpret_cast<char*>(hdr.data()), (std::streamsize)ACM_HEADER_SIZE);
    if ((size_t)fin.gcount() < (size_t)ACM_HEADER_SIZE)
        return fail("acm 头部不完整（不足 1024 字节）");
    AcmHeader h; NetSpec spec; std::string sig; std::string e;
    if (!decodeAcmHeader(hdr, h, spec, sig, &e)) return fail(e);
    out = spec;
    return true;
}

// T4 统一入口：读侧恒按魔数自动识别。
// R1：显式 .acm 后缀但魔数不符（=损坏）→ 写 err 并返回 false（不回退 txt）。
static bool peekModelDims(const std::string& path, NetSpec& out,
                          std::string* err = nullptr) {
    const ModelFormat fmt = probeModelFormat(path);
    if (fmt == ModelFormat::Acm) return peekModelDimsAcm(path, out, err);
    if (fmt == ModelFormat::Unknown) {
        if (err) *err = "后缀为 .acm 但魔数不符（文件损坏或并非 acm 格式）";
        return false;
    }
    return peekModelDimsTxt(path, out, err);
}

// ================= 模型维度校验（集成/加载时用） =================
// 阶段0+1：由“校验等于编译期常量”改为“校验与运行时规格 g_net 自洽”。
// 若文件里的卷积形状与 g_net 不符，flatten/scatter 会越界，必须先拒绝。
static bool checkModelDims(const std::string& path, const FullyConnected& fc) {
    bool ok = (fc.outDim == g_net.classes && fc.inDim == g_net.flat);
    for (int i = 0; i < g_net.depth && ok; ++i) {
        const K4& K = KERNELS_L[i];
        const LayerSpec& L = g_net.layer[i];
        ok = (int)K.size() == L.outC &&
             (int)K[0].size() == L.inC &&
             (int)K[0][0].size() == L.kH &&
             (int)K[0][0][0].size() == L.kW &&
             (int)BIAS_L[i].size() == L.outC;
        if (ok && L.res == 2) {
            // 投影段形状须与 g_net 推导一致（outC × srcC × 1 × 1 + outC）
            const int SC = (L.resFrom >= 0 && L.resFrom < i) ? g_net.layer[L.resFrom].outC : -1;
            ok = (SC > 0) &&
                 (int)PROJ_K_L[i].size() == L.outC &&
                 (int)PROJ_K_L[i][0].size() == SC &&
                 (int)PROJ_K_L[i][0][0].size() == 1 &&
                 (int)PROJ_K_L[i][0][0][0].size() == 1 &&
                 (int)PROJ_B_L[i].size() == L.outC;
        }
    }
    if (!ok) {
        std::cerr << "模型维度不符: " << path << "（规格 ";
        for (int i = 0; i < g_net.depth; ++i) {
            const LayerSpec& L = g_net.layer[i];
            std::cerr << "Conv" << (i + 1) << " " << L.outC << "x" << L.inC
                << "x" << L.kH << "x" << L.kW << "，";
        }
        std::cerr << "FC " << g_net.classes << "x" << g_net.flat << "）\n";
    }
    return ok;
}

// ================= BN 与 CLI 一致性检测（P0） =================
// 在 peekModelDims 之后、buildKernels 之前调用。返回 false 表示应终止（已打印错误）。
//   modelBn：模型文件声明的 BN（无 #SIG / 无 BN 字段的旧模型归一化为 false）
//   cliBn  ：命令行是否传 --BN/--bn
//   mode   ：gen/bkc/test/GUI/(单图)，用于错误信息
static bool checkBnCliAgreement(bool modelBn, bool cliBn, const char* mode) {
    const char* m = (mode && *mode) ? mode : "运行";
    if (modelBn && !cliBn) {
        std::cerr << m << ": 模型为 BN:ON，但未传 --BN。BN 模型必须显式加 --BN 才能加载/运行\n";
        return false;
    }
    if (!modelBn && cliBn) {
        std::cerr << m << ": 传了 --BN，但模型为 BN:OFF。BN 只能在 gen 时决定，"
                     "不能给非 BN 模型启用 --BN\n";
        return false;
    }
    return true;
}

// ================= 逗号分隔的模型路径解析（test 集成用） =================
// "a.txt,b.txt" -> {"a.txt","b.txt"}；去除每段首尾空白；忽略空段。
// 路径本身可含空格，仅按逗号拆分，故不会误分带空格的单个路径。
static std::vector<std::string> splitModelList(const std::string& s) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = (comma == std::string::npos)
            ? s.substr(pos) : s.substr(pos, comma - pos);
        size_t b = tok.find_first_not_of(" \t\r\n");
        size_t e = tok.find_last_not_of(" \t\r\n");
        if (b != std::string::npos) out.push_back(tok.substr(b, e - b + 1));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

// ================= 逗号分隔的浮点解析（--weights 用） =================
static void parseFloatList(const std::string& s, std::vector<float>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t b = tok.find_first_not_of(" \t\r\n");
        size_t e = tok.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        out.push_back(std::stof(tok.substr(b, e - b + 1)));
    }
}

// ============================================================================
// 集成融合（test）—— 六种融合模式
//   probs : M*N*C 概率张量（C=g_net.classes），索引 ((size_t)m*N + i)*C + c
//   w     : 每模型权重（长度不足按 1 补，超出忽略）
//   temp  : 概率温度缩放（p^(1/temp) 后逐模型归一化）
//   返回第 i 个样本的预测类
// ============================================================================
enum FuseMode { FUSE_SOFT, FUSE_LOGIT, FUSE_GEOM, FUSE_VOTE, FUSE_GATE, FUSE_WBEST };

static const char* fuseModeName(FuseMode m) {
    switch (m) {
    case FUSE_LOGIT: return "logit";
    case FUSE_GEOM:  return "geom";
    case FUSE_VOTE:  return "vote";
    case FUSE_GATE:  return "gate";
    case FUSE_WBEST: return "wbest";
    default:         return "soft";
    }
}

static int fusePredictSample(const float* probs, int N, int M, int i,
    const std::vector<float>& w, FuseMode mode, float margin, float temp)
{
    const int C = g_net.classes;      // 运行时类别数（替代硬编码 CNN_CLASSES）
    if (M <= 0) return 0;
    if (temp <= 0.0f) temp = 1.0f;

    // 逐模型取出本样本概率，做温度缩放并重新归一化为概率分布
    std::vector<float> p((size_t)M * C, 0.0f);
    for (int m = 0; m < M; ++m) {
        const float* src = probs + ((size_t)m * N + i) * C;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = src[c];
            if (v < 0.0f) v = 0.0f;
            float sv = (temp == 1.0f) ? v : std::pow(v, 1.0f / temp);
            p[(size_t)m * C + c] = sv;
            sum += sv;
        }
        if (sum > 0.0f)
            for (int c = 0; c < C; ++c) p[(size_t)m * C + c] /= sum;
    }

    // 每个模型的 argmax
    std::vector<int> argM(M, 0);
    for (int m = 0; m < M; ++m) {
        const float* pm = p.data() + (size_t)m * C;
        argM[m] = argmaxIndex(pm, C);
    }

    // 权重取值（缺省 1）
    auto wof = [&](int m) -> double {
        return (m < (int)w.size()) ? (double)w[m] : 1.0;
    };

    switch (mode) {
    case FUSE_WBEST: {                       // 直接取权重最大模型的 argmax
        int best = 0;
        for (int m = 1; m < M; ++m) if (wof(m) > wof(best)) best = m;
        return argM[best];
    }
    case FUSE_VOTE: {                        // 每模型一票，多数票；平票取权重最高模型
        std::vector<int> cnt(C, 0);          // 长度 = 运行时类别数
        for (int m = 0; m < M; ++m) cnt[argM[m]]++;
        int best = 0;
        for (int c = 1; c < C; ++c) {
            if (cnt[c] > cnt[best]) { best = c; continue; }
            if (cnt[c] == cnt[best]) {
                double wc = -1.0, wb = -1.0;
                for (int m = 0; m < M; ++m) {
                    if (argM[m] == c && wof(m) > wc) wc = wof(m);
                    if (argM[m] == best && wof(m) > wb) wb = wof(m);
                }
                if (wc > wb) best = c;
            }
        }
        return best;
    }
    case FUSE_GATE:                          // 3 模型场景门控
        if (M == 3) {
            int a0 = argM[0], a1 = argM[1], a2 = argM[2];
            if (a0 == a1 && a1 == a2) return a0;      // 三票一致
            int maj = -1, solo = -1;
            if (a0 == a1) { maj = a0; solo = a2; }
            else if (a0 == a2) { maj = a0; solo = a1; }
            else if (a1 == a2) { maj = a1; solo = a0; }
            if (maj >= 0 && solo >= 0) {              // 2:1
                float soloP = 0.0f, majP = 0.0f;
                int nc = 0, nm = 0;
                for (int m = 0; m < M; ++m) {
                    if (argM[m] == solo) { soloP += p[(size_t)m * C + solo]; ++nc; }
                    if (argM[m] == maj)  { majP  += p[(size_t)m * C + maj];  ++nm; }
                }
                if (nc) soloP /= nc;
                if (nm) majP /= nm;
                if (soloP > majP + margin) return solo; // 独醒者够自信则反选
                return maj;
            }
            int best = 0;                              // 三票分散
            for (int m = 1; m < M; ++m) if (wof(m) > wof(best)) best = m;
            return argM[best];
        }
        // M != 3 时退化为加权 soft（fallthrough）
    case FUSE_SOFT:
    case FUSE_LOGIT:
    case FUSE_GEOM:
    default: {
        std::vector<double> acc(C, 0.0);
        bool useLog = (mode == FUSE_LOGIT || mode == FUSE_GEOM);
        for (int m = 0; m < M; ++m) {
            const float* pm = p.data() + (size_t)m * C;
            double wm = wof(m);
            for (int c = 0; c < C; ++c)
                acc[c] += wm * (useLog ? std::log((double)pm[c] + 1e-9)
                                       : (double)pm[c]);
        }
        int best = argmaxIndex(acc.data(), C);
        return best;
    }
    }
}

// 加权平均“某类”概率（--outerror 置信度显示用）
static float modelWeightedProb(const float* probs, int N, int M, int i,
    const std::vector<float>& w, int cls)
{
    float acc = 0.0f, wsum = 0.0f;
    const int C = g_net.classes;   // 运行时类别数（概率张量步长）
    for (int m = 0; m < M; ++m) {
        float wm = (m < (int)w.size()) ? w[m] : 1.0f;
        acc += wm * probs[((size_t)m * N + i) * C + cls];
        wsum += wm;
    }
    return (wsum > 0.0f) ? acc / wsum : 0.0f;
}

// 把 log 概率累加器（N*M*C，C=g_net.classes）按变换数 T 平均后
// softmax 回概率张量 allProb。softmax(mean log p) 等价于 softmax(mean logit)。
static void logitAccToProb(const std::vector<double>& logitAcc, int N, int M,
    int T, std::vector<float>& allProb)
{
    const int C = g_net.classes;   // 运行时类别数
    allProb.assign((size_t)N * M * C, 0.0f);
    double invT = (T > 0) ? (1.0 / (double)T) : 1.0;
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < N; ++i) {
            const double* acc = logitAcc.data() + ((size_t)m * N + i) * C;
            float* dst = allProb.data() + ((size_t)m * N + i) * C;
            double mx = acc[0];
            for (int c = 1; c < C; ++c) if (acc[c] > mx) mx = acc[c];
            double sum = 0.0;
            for (int c = 0; c < C; ++c) {
                double e = std::exp((acc[c] - mx) * invT);
                dst[c] = (float)e;
                sum += e;
            }
            if (sum > 0.0)
                for (int c = 0; c < C; ++c) dst[c] = (float)(dst[c] / sum);
            else
                dst[0] = 1.0f;
        }
}

// ============================================================================
// TTA（测试时增强）—— 对“已预处理”的 32×32 二值图做轻微几何变换
//   关键：只做几何变换，绝不再次调用 preprocessImage（否则会抵消变换）
// ============================================================================
static void ttaRotate(ConvMap& img, float deg) {
    if (deg == 0.0f) return;
    double rad = (double)deg * CNN_PI / 180.0;
    float cosT = (float)std::cos(rad);
    float sinT = (float)std::sin(rad);
    float cx = (img.W - 1) * 0.5f;
    float cy = (img.H - 1) * 0.5f;
    ConvMap out(img.C, img.H, img.W);
    for (int oh = 0; oh < img.H; ++oh)
        for (int ow = 0; ow < img.W; ++ow) {
            float x = (float)ow - cx;
            float y = (float)oh - cy;
            float xs = (cosT * x + sinT * y) + cx;
            float ys = (-sinT * x + cosT * y) + cy;
            int sx = (int)std::lround(xs);
            int sy = (int)std::lround(ys);
            if (sx < 0 || sx >= img.W || sy < 0 || sy >= img.H) continue;
            for (int c = 0; c < img.C; ++c)
                if (prepIsForeground(img.data[c][sy][sx]))
                    out.data[c][oh][ow] = 1.0f;
        }
    img = out;
    finalizeBinary(img);
}

struct TTATransform { int dx; int dy; float deg; };

// 固定变换序列（由温和到更强，t0 恒为原图）
static const TTATransform* ttaCandidates(int& nOut) {
    static const TTATransform cand[] = {
        { 0,  0,   0.0f },   // t0：原图
        { 1,  0,   0.0f },   // 右移 1px
        {-1,  0,   0.0f },   // 左移 1px
        { 0,  0,   8.0f },   // +8°
        { 0,  0,  -8.0f },   // -8°
        { 2,  0,   0.0f },   // 右移 2px
        {-2,  0,   0.0f },   // 左移 2px
        { 0,  0,  12.0f },   // +12°
        { 0,  0, -12.0f },   // -12°
        { 0,  1,   0.0f },   // 下移 1px
        { 0, -1,   0.0f },   // 上移 1px
    };
    nOut = (int)(sizeof(cand) / sizeof(cand[0]));
    return cand;
}

static int ttaClampK(int k) {
    int n = 0; ttaCandidates(n);
    if (k < 1) k = 1;
    if (k > n) k = n;
    return k;
}

// 批量版：把扁平数据 data(N*1024) 按第 t 个变换生成 outData(N*1024)
static void makeTTADataForTransform(const std::vector<float>& data, int N,
    int t, std::vector<float>& outData)
{
    int n = 0; const TTATransform* cand = ttaCandidates(n);
    if (t < 0) t = 0;
    if (t >= n) t = n - 1;
    // L3：输入尺寸改用运行时规格 g_net.inH/inW（当前 peek 恒为 32×32，行为不变）
    const int H = g_net.inH, W = g_net.inW;
    const size_t HW = (size_t)H * W;
    outData.resize((size_t)N * HW);
    for (int i = 0; i < N; ++i) {
        ConvMap img(1, H, W);
        const float* src = data.data() + (size_t)i * HW;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                img.data[0][h][w] = src[h * W + w];
        if (cand[t].dx != 0 || cand[t].dy != 0) prepShift(img, cand[t].dx, cand[t].dy);
        if (cand[t].deg != 0.0f) ttaRotate(img, cand[t].deg);
        finalizeBinary(img);
        float* dst = outData.data() + (size_t)i * HW;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                dst[h * W + w] = img.data[0][h][w];
    }
}

// ================= 训练样本（含域标记） =================
// domain: 0 = 主目录(path)，1 = --data2 目录
struct Sample {
    std::string path;
    int label = -1;
    int domain = 0;
};

// 扫描单目录收集样本（只记路径/标签/域，不读图）
static void collectSamples(const std::string& dir, int domain,
    std::vector<Sample>& out)
{
    std::vector<std::string> files = listTxtFiles(dir);
    for (auto& f : files) {
        int label = getLabelFromPath(f);
        if (label < 0) continue;
        Sample s;
        s.path = f;
        s.label = label;
        s.domain = domain;
        out.push_back(s);
    }
}

// ================= 训练集构建（域均衡 / 全量投喂） =================
// 无 data2：直接返回主域全部样本（顺序不变，完全复现旧行为，不减样不重排）。
// 有 data2：
//   fullMix=false（balanced，默认）：按 (domain,label) 分层，取所有非空层
//          数量的最小值作为每层目标 target；每层随机抽取 target 个后合并
//          打乱。这样两个域、10 个类别贡献完全对等，避免 MNIST(60000)
//          淹没 optdigits(1934)。
//   fullMix=true （--mix full）：全量投喂，合并两域全部样本，不做任何
//          下采样/均衡，仅打乱顺序。样本规模 = 域0 + 域1（如 1934+60000）。
static std::vector<Sample> buildTrainSet(const std::string& dir0,
    const std::string& dir1,
    unsigned seed,
    bool fullMix)
{
    std::vector<Sample> s0;
    collectSamples(dir0, 0, s0);

    if (dir1.empty()) {
        std::cout << "训练文件数: " << s0.size() << "（单域，未做均衡）\n";
        return s0; // 完全复现旧行为
    }

    std::vector<Sample> s1;
    collectSamples(dir1, 1, s1);

    // ---------- 全量投喂：合并两域全部样本，绝不减样 ----------
    if (fullMix) {
        std::cout << "混合训练(全量投喂)：域0=" << s0.size()
            << "，域1=" << s1.size()
            << "，合计=" << (s0.size() + s1.size())
            << "（未做下采样/均衡）\n";
        std::vector<Sample> out;
        out.reserve(s0.size() + s1.size());
        out.insert(out.end(), s0.begin(), s0.end());
        out.insert(out.end(), s1.begin(), s1.end());
        std::mt19937 rng(seed);
        std::shuffle(out.begin(), out.end(), rng);
        std::cout << "全量投喂训练样本数: " << out.size() << "\n";
        return out;
    }

    // ---------- 域均衡分层采样（默认行为，保持向后兼容） ----------
    const int K = g_net.classes;   // 运行时类别数（替代硬编码 10）
    std::vector<std::vector<int>> cnt(2, std::vector<int>(K, 0));
    for (auto& s : s0) if (s.label >= 0 && s.label < K) cnt[0][s.label]++;
    for (auto& s : s1) if (s.label >= 0 && s.label < K) cnt[1][s.label]++;

    int target = INT_MAX;
    for (int d = 0; d < 2; ++d)
        for (int l = 0; l < K; ++l)
            if (cnt[d][l] > 0 && cnt[d][l] < target) target = cnt[d][l];
    if (target == INT_MAX) target = 0;

    std::cout << "混合训练：域0=" << s0.size() << "，域1=" << s1.size()
        << "，各层目标=" << target << "\n";

    std::vector<Sample> out;
    std::mt19937 rng(seed);

    // 逐域逐类随机下采样到 target
    auto pick = [&](std::vector<Sample>& src) {
        std::vector<std::vector<Sample>> buckets(K);
        for (auto& s : src)
            if (s.label >= 0 && s.label < K) buckets[s.label].push_back(s);
        for (int l = 0; l < K; ++l) {
            std::vector<Sample>& b = buckets[l];
            std::shuffle(b.begin(), b.end(), rng);
            int take = (std::min)((int)b.size(), target);
            for (int i = 0; i < take; ++i) out.push_back(b[i]);
        }
    };
    pick(s0);
    pick(s1);

    std::shuffle(out.begin(), out.end(), rng);
    std::cout << "均衡后训练样本数: " << out.size() << "\n";
    return out;
}

// 把一张 32×32 二值图压平为 1024 维追加到 data
static void pushFlat(const ConvMap& img, int label,
    std::vector<float>& data, std::vector<int>& labels)
{
    for (int h = 0; h < img.H; ++h)
        for (int w = 0; w < img.W; ++w)
            data.push_back(img.data[0][h][w]);
    labels.push_back(label);
}

// ================= 独立验证集评估（--val，阶段4.4） =================
// 目录规范化：取绝对路径 + 全小写 + 去尾部斜杠，用于判定 val 是否等于训练目录。
static std::string normalizeDir(const std::string& d) {
    char buf[MAX_PATH];
    DWORD n = GetFullPathNameA(d.c_str(), MAX_PATH, buf, nullptr);
    std::string s = (n > 0 && n < MAX_PATH) ? std::string(buf) : d;
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    while (!s.empty() && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s;
}

// 加载验证集：与 test CPU 路径同源（listTxtFiles → getLabelFromPath →
// loadImagePreprocessed(g_prep) → pushFlat）；不做任何增强，图像预处理只做一次。
// 返回 false 表示没有可用验证样本（此时 --val 等价于只打印告警、不产出 val 数值）。
static bool loadValSet(const std::string& dir) {
    g_valData.clear();
    g_valLabels.clear();
    std::vector<std::string> files = listTxtFiles(dir);
    for (auto& f : files) {
        int label = getLabelFromPath(f);
        if (label < 0) continue;
        ConvMap input(1, 32, 32);
        if (!loadImagePreprocessed(f, input)) continue;
        pushFlat(input, label, g_valData, g_valLabels);
    }
    return !g_valLabels.empty();
}

// 在当前全局权重（KERNELS_L/BIAS_L/BN_* 与 fc）上对验证集算一次准确率。
// 路径与 test CPU 前向完全一致：1024 维输入 → forwardConv → flatten → fc.forward → softmax → argmax。
// 通过 outCorrect/outTotal 回传计数，便于 stdout 打印 (correct/total)。无样本返回 -1。
static float evalValAccuracy(FullyConnected& fc, int* outCorrect = nullptr,
                             int* outTotal = nullptr) {
    if (g_valLabels.empty()) return -1.0f;
    int correct = 0, total = 0;
    std::vector<float> prob;
    for (size_t i = 0; i < g_valLabels.size(); ++i) {
        const float* src = g_valData.data() + i * 1024;
        ConvMap input(1, 32, 32);
        for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w)
                input.data[0][h][w] = src[h * 32 + w];
        ConvMap p2 = forwardConv(input);
        std::vector<float> flat = flatten(p2);
        prob = softmax(fc.forward(flat));
        int pred = 0;
        for (int c = 1; c < (int)prob.size(); ++c)
            if (prob[c] > prob[pred]) pred = c;
        if (pred == g_valLabels[i]) ++correct;
        ++total;
    }
    if (outCorrect) *outCorrect = correct;
    if (outTotal) *outTotal = total;
    return total > 0 ? (float)correct / total : -1.0f;
}

// 评估验证集 + 打印 + 更新最优模型（<model>.best）。返回准确率（无样本 -1）。
// 说明：仅「严格变好」才覆盖 best；保存 best 只读当前全局权重，不改变训练状态。
static float runValEvalAndBest(FullyConnected& fc, int epoch,
                               const std::string& bestPath) {
    int correct = 0, total = 0;
    float acc = evalValAccuracy(fc, &correct, &total);
    if (acc < 0.0f) {
        std::cout << "val acc: 无可用验证样本\n";
        return acc;
    }
    std::cout << "val acc " << acc << " (" << correct << "/" << total << ")\n";
    if (g_saveBest && acc > g_bestValAcc) {
        g_bestValAcc = acc;
        if (saveModel(bestPath, fc)) {
            std::cout << "saved best " << bestPath << " (val acc " << acc << ")\n";
            logCheckpoint(epoch, bestPath);
        }
    }
    return acc;
}

// ================= 数据校验（--checkdata / --strict-data，阶段4.4） =================
// 逐个 txt 文件检查：文件名首字符为数字；内容为 32 行 × 32 列、仅含 '0'/'1'、非空。
// 返回结构：label（首字符数字，否则 -1）、illegal、reason（非法原因，中文）。
struct DataCheckResult {
    int         label = -1;
    bool        illegal = false;
    std::string reason;
};

static std::string baseName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// --epsave 默认目录用：取 basename 后去掉最后一个 '.' 的扩展名。
//   E:\CNN\models\b4_1.txt -> "b4_1"；无扩展名保持原样；结果为空则回退 "model"。
static std::string modelBaseName(const std::string& path) {
    std::string n = baseName(path);
    size_t dot = n.find_last_of('.');
    if (dot != std::string::npos) n = n.substr(0, dot);
    return n.empty() ? std::string("model") : n;
}

// --epsave 用：逐段创建目录（处理盘符 "X:"；ERROR_ALREADY_EXISTS 视为成功）。
// 不使用 <filesystem>（Debug 配置未启用 C++17）。仅创建，不改动其它逻辑。
static bool ensureDirectory(const std::string& dir) {
    if (dir.empty()) return false;
    std::string d = dir;
    for (char& c : d) if (c == '/') c = '\\';   // 统一分隔符
    std::string cur;
    size_t i = 0;
    if (d.size() >= 2 && d[1] == ':') { cur = d.substr(0, 2); i = 2; }   // 盘符前缀
    while (i < d.size()) {
        if (d[i] == '\\') { ++i; continue; }
        size_t j = i;
        while (j < d.size() && d[j] != '\\') ++j;
        std::string seg = d.substr(i, j - i);
        if (cur.empty()) cur = seg;
        else cur += (cur.back() == '\\') ? seg : ("\\" + seg);
        // 跳过纯盘符段（"X:"）；其余逐段创建
        if (cur.back() != ':') {
            if (!CreateDirectoryA(cur.c_str(), nullptr)) {
                DWORD e = GetLastError();
                if (e != ERROR_ALREADY_EXISTS) return false;
            }
        }
        i = j;
    }
    return true;
}

static DataCheckResult inspectDataFile(const std::string& path) {
    DataCheckResult r;
    std::string name = baseName(path);
    if (name.empty() || name[0] < '0' || name[0] > '9') {
        r.illegal = true;
        r.reason = "文件名首字符非数字";
        return r;
    }
    r.label = name[0] - '0';

    std::ifstream fin(path);
    if (!fin) { r.illegal = true; r.reason = "打开失败"; return r; }

    std::string line;
    int  rows = 0;
    bool badWidth = false, badChar = false;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // 兼容 CRLF
        if (line.empty()) continue;                                  // 与 loadImage 一致：跳过空行
        ++rows;
        if (line.size() != 32) { badWidth = true; continue; }
        for (int w = 0; w < 32; ++w) {
            char c = line[w];
            if (c != '0' && c != '1') { badChar = true; break; }
        }
    }
    if (rows == 0)        { r.illegal = true; r.reason = "空文件"; return r; }
    if (rows != 32)       { r.illegal = true; r.reason = "行数 != 32（实际 " + std::to_string(rows) + "）"; return r; }
    if (badWidth)         { r.illegal = true; r.reason = "存在非 32 列的行"; return r; }
    if (badChar)          { r.illegal = true; r.reason = "含非 0/1 字符"; return r; }
    return r;
}

// 扫描目录并打印报告：文件数 / 每类计数 / 非法文件清单。
// 返回非法文件个数；strict=true 且存在非法时额外在 stderr 报错（调用方据此非零退出）。
static int checkDataDir(const std::string& dir, bool strict) {
    std::vector<std::string> files = listTxtFiles(dir);
    int counts[10] = { 0 };
    std::vector<std::pair<std::string, std::string>> illegal;
    for (auto& f : files) {
        DataCheckResult r = inspectDataFile(f);
        if (r.illegal) illegal.push_back(std::make_pair(f, r.reason));
        else if (r.label >= 0 && r.label < 10) ++counts[r.label];
    }
    std::cout << "[数据校验] 目录: " << dir << "\n";
    std::cout << "[数据校验] 文件数: " << files.size() << "\n";
    std::cout << "[数据校验] 合法: " << (files.size() - illegal.size())
        << "  非法: " << illegal.size() << "\n";
    std::cout << "[数据校验] 每类计数:";
    for (int i = 0; i < 10; ++i) std::cout << " " << i << ":" << counts[i];
    std::cout << "\n";
    if (!illegal.empty()) {
        std::cout << "[数据校验] 非法文件:\n";
        for (auto& p : illegal)
            std::cout << "  - " << p.first << ": " << p.second << "\n";
    }
    if (strict && !illegal.empty())
        std::cerr << "[数据校验] --strict-data：发现 " << illegal.size() << " 个非法文件，退出\n";
    return (int)illegal.size();
}

// ================= 并行加载：线程本地结果 + 分段工作函数 =================
// 每个线程独立处理 samples 的一个下标区间，写自己的本地缓冲，主线程 join
// 后按线程顺序无锁合并，避免任何共享写竞争。
struct LoadResult {
    std::vector<float> data;
    std::vector<int>   labels;
    size_t okOriginal = 0; // 成功读入并预处理的原始图数
    size_t produced   = 0; // 实际产出的样本总数（原图 + 增强副本）
    bool   oom        = false;
};

// 处理 samples[begin, end)：每张原图串行地产生 1 份原图 + aug.copies 份增强副本。
// 每样本用 (seedBase, i) 派生独立 RNG，与线程划分无关 → 固定 seed 可复现。
static void loadRangeWorker(size_t begin, size_t end,
    const std::vector<Sample>& samples, const AugConfig& aug,
    unsigned seedBase, LoadResult& out,
    std::atomic<size_t>& done, std::mutex& printMtx, size_t total)
{
    const size_t copies = aug.enable ? ((size_t)aug.copies + 1) : 1;
    try {
        out.data.reserve((end - begin) * copies * 1024);
        out.labels.reserve((end - begin) * copies);
    }
    catch (const std::bad_alloc&) { out.oom = true; return; }

    std::mt19937 rng;
    const size_t step = std::max<size_t>(1, total / 20); // 每约 5% 打印一次进度
    for (size_t i = begin; i < end; ++i) {
        const Sample& s = samples[i];
        ConvMap input(1, 32, 32);
        if (loadImagePreprocessed(s.path, input)) {
            // 每样本独立 RNG：种子只依赖 (seedBase, 样本下标)，与线程无关
            std::seed_seq seq{ seedBase, (unsigned)i, (unsigned)(i >> 16) };
            rng.seed(seq);
            try {
                pushFlat(input, s.label, out.data, out.labels);
                ++out.produced; ++out.okOriginal;
                if (aug.enable) {
                    for (int k = 0; k < aug.copies; ++k) {
                        ConvMap a = input;
                        augmentRandom(a, aug, rng);
                        pushFlat(a, s.label, out.data, out.labels);
                        ++out.produced;
                    }
                }
            }
            catch (const std::bad_alloc&) { out.oom = true; return; }
        }
        else {
            std::lock_guard<std::mutex> lk(printMtx);
            std::cerr << "跳过无效样本: " << s.path << "\n";
        }

        size_t n = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % step == 0 || n == total) {
            std::lock_guard<std::mutex> lk(printMtx);
            std::cout << "\r[加载] " << n << " / " << total
                << " (" << (100 * n / total) << "%)" << std::flush;
        }
    }
}

// ================= GPU 数据准备：读图 + 预处理 (+ 离线扩增) =================
// 单域且 --aug 关闭时，顺序与规模同旧 loadAllData，行为完全一致。
// --aug 开启时每张原图额外生成 aug.copies 个独立随机增强副本。
static bool loadAllDataSamples(const std::vector<Sample>& samples,
    const AugConfig& aug,
    std::mt19937& rng,
    std::vector<float>& data,
    std::vector<int>& labels,
    int threads,
    bool timeIt,
    unsigned seedBase)
{
    data.clear();
    labels.clear();

    // 每张原图产出 (aug.copies + 1) 份（原图 + 增强副本）。
    // copies 用 size_t 计算，避免 `int` 加法在极端值下溢出；30 万级样本
    // ×1024 float 约 1.2 GB，精确 reserve 比 vector 几何扩容（峰值 ~2x）更省内存。
    size_t copies = aug.enable ? ((size_t)aug.copies + 1) : 1;
    const size_t expectSamples = samples.size() * copies; // size_t 乘法，无 int 截断

    if (expectSamples > ((size_t)-1) / 1024) {
        std::cerr << "样本规模过大，超出可寻址范围\n";
        return false;
    }

    // ---------- legacy 单线程分支（--threads 1）：行为与旧版完全一致 ----------
    if (threads <= 1) {
        try {
            data.reserve(expectSamples * 1024);
            labels.reserve(expectSamples);
        }
        catch (const std::bad_alloc&) {
            std::cerr << "内存不足：无法为 " << expectSamples << " 个样本预分配 "
                << (expectSamples * 1024 * sizeof(float) / (1024.0 * 1024.0))
                << " MB 训练数据\n";
            return false;
        }

        for (auto& s : samples) {
            ConvMap input(1, 32, 32);
            if (!loadImagePreprocessed(s.path, input)) continue;

            pushFlat(input, s.label, data, labels); // 原图

            if (aug.enable) {
                for (int k = 0; k < aug.copies; ++k) {
                    ConvMap a = input;
                    augmentRandom(a, aug, rng);
                    pushFlat(a, s.label, data, labels);
                }
            }
        }

        std::cout << "train files: " << samples.size()
            << " -> samples: " << labels.size();
        if (aug.enable)
            std::cout << "（每张原图 + " << aug.copies << " 个增强副本）";
        std::cout << "\n";
        return !labels.empty();
    }

    // ---------- 并行分支（--threads >= 2 / auto） ----------
    const size_t N = samples.size();
    const int T = threads;

    auto tParallel0 = std::chrono::steady_clock::now();
    std::cout << "并行加载: threads=" << T << "，原始样本=" << N << "\n";

    std::vector<LoadResult> results((size_t)T);
    std::atomic<size_t> done{ 0 };
    std::mutex printMtx;

    // 静态分块：下标域 [0,N) 均匀切成 T 段，逐线程独立处理。
    std::vector<std::thread> workers;
    workers.reserve((size_t)T);
    for (int t = 0; t < T; ++t) {
        const size_t begin = N * (size_t)t / (size_t)T;
        const size_t end = N * (size_t)(t + 1) / (size_t)T;
        workers.emplace_back([&, t, begin, end]() {
            loadRangeWorker(begin, end, samples, aug, seedBase,
                results[(size_t)t], done, printMtx, N);
        });
    }
    for (auto& w : workers) w.join();

    // 汇总各线程本地结果
    size_t totalProduced = 0, totalOk = 0;
    bool anyOom = false;
    for (auto& r : results) {
        totalProduced += r.produced;
        totalOk += r.okOriginal;
        if (r.oom) anyOom = true;
    }

    if (anyOom) {
        std::cerr << "内存不足：无法为 " << expectSamples << " 个样本预分配 "
            << (expectSamples * 1024 * sizeof(float) / (1024.0 * 1024.0))
            << " MB 训练数据\n";
        return false;
    }

    // 主线程按线程顺序合并（无锁）：先精确分配，再逐线程 copy。
    try {
        data.resize(totalProduced * 1024);
        labels.resize(totalProduced);
    }
    catch (const std::bad_alloc&) {
        std::cerr << "内存不足：无法为 " << expectSamples << " 个样本预分配 "
            << (expectSamples * 1024 * sizeof(float) / (1024.0 * 1024.0))
            << " MB 训练数据\n";
        return false;
    }

    size_t dOff = 0, lOff = 0;
    for (auto& r : results) {
        if (!r.data.empty()) {
            std::copy(r.data.begin(), r.data.end(), data.begin() + dOff);
            dOff += r.data.size();
        }
        if (!r.labels.empty()) {
            std::copy(r.labels.begin(), r.labels.end(), labels.begin() + lOff);
            lOff += r.labels.size();
        }
        // 合并后立即释放线程本地缓冲，降低峰值内存
        std::vector<float>().swap(r.data);
        std::vector<int>().swap(r.labels);
    }

    std::cout << "\ntrain files: " << N
        << " -> samples: " << labels.size();
    if (aug.enable)
        std::cout << "（每张原图 + " << aug.copies << " 个增强副本）";
    std::cout << "，有效原图 " << totalOk << "/" << N << "\n";
    if (timeIt) {
        double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tParallel0).count();
        std::cout << "并行加载耗时: " << formatDur(sec)
            << "（threads=" << T << "）\n";
    }
    return !labels.empty();
}

// ================= 显存占用估算（启动时报告训练规模用） =================
// 与 gpu_kernels.cuh 的缓冲布局一致（方案甲）：
//   常驻：d_data[N*1024] + d_allLabel[N] + d_index[N]
//   批内：每样本 121108 float（d_input/d_c1/d_r1/d_p1/d_c2/d_r2/d_p2/
//         d_logits/d_prob/d_dflat/d_dc2/d_dp1/d_dc1）+ d_label(1 int)
//   固定：d_params[42378] + d_grad[42378] + d_velocity[42378]
static void printGpuMemoryEstimate(size_t N, int batch) {
    // 阶段0+1：用 g_net 推导各中间张量大小；默认 32/64 下与旧常量一致
    // （perBatchFloats=121108，param 总量=42378）。
    const size_t inSize = (size_t)g_net.inH * g_net.inW;
    const size_t perSample     = inSize * sizeof(float) + 2 * sizeof(int);
    // 逐层：d_c/d_r/d_dc 各 outC*H*W，d_p/d_dp 各 outC*(H/2)*(W/2)；BN 时另加 d_b/d_db。
    size_t perLayer = 0;
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        const size_t act  = (size_t)L.outC * (size_t)L.H * (size_t)L.W;
        const size_t poolH = L.pool ? (size_t)(L.H / 2) : (size_t)L.H;
        const size_t poolW = L.pool ? (size_t)(L.W / 2) : (size_t)L.W;
        const size_t pool = (size_t)L.outC * poolH * poolW;
        perLayer += 3 * act + 2 * pool;
        if (g_net.bn) perLayer += 2 * act;   // d_b, d_db
    }
    const size_t perBatchFloats =
        inSize
        + perLayer
        + 2 * (size_t)g_net.classes;    // d_logits, d_prob
    const size_t perBatchBytes = perBatchFloats * sizeof(float) + (size_t)1 * sizeof(int);
    const size_t dataBytes  = N * perSample;
    const size_t batchBytes = (size_t)batch * perBatchBytes;
    const size_t paramBytes = (size_t)3 * (size_t)g_net.total * sizeof(float);
    const size_t total      = dataBytes + batchBytes + paramBytes;
    const double MiB = 1024.0 * 1024.0;
    std::cout << "预计显存占用: 常驻全量数据 " << (dataBytes / MiB) << " MB"
        << " + 批内缓冲(batch=" << batch << ") " << (batchBytes / MiB) << " MB"
        << " + 参数 " << (paramBytes / MiB) << " MB"
        << " ≈ " << (total / MiB) << " MB ("
        << (total / (MiB * 1024.0)) << " GB)\n";
}

// epoch end: sync weights back to globals and save intermediate model
static void gpuEpochEndCallback(const float* p, int epoch,
                                float loss, float acc,
                                float lr1, float lr2, float lrfc,
                                double epochSec, void* userData) {
    FullyConnected* fc = (FullyConnected*)userData;
    scatterParams(p, *fc);
    // 阶段4.4：--val 时在验证集上评估（scatter 后全局权重即为本 epoch 末权重）。
    float valAcc = -1.0f;
    bool  hasVal = false;
    if (g_valEnabled) {
        valAcc = runValEvalAndBest(*fc, epoch, g_bestPath);
        hasVal = (valAcc >= 0.0f);
    }
    // 阶段4.3：--log 时先落本 epoch 指标（追加 + flush）
    logEpochMetrics(epoch, loss, acc, lr1, lr2, lrfc, epochSec, valAcc, hasVal);
    if (epoch % 10 == 9) {
        std::string epPath = withModelExt("model_epoch" + std::to_string(epoch), currentSaveFormat());
        saveModel(epPath, *fc);
        std::cout << "saved " << epPath << "\n";
        logCheckpoint(epoch, epPath);   // 阶段4.3：记录存档元数据
    }
    // 功能二：--epsave 每轮存档（与上面每 10 轮存档并存、互不影响）。
    if (g_epsaveEnabled && g_epsn > 0 && (epoch % g_epsn == 0)) {
        std::string epPath = g_epSaveDir + withModelExt("ep_" + std::to_string(epoch), currentSaveFormat());
        saveModel(epPath, *fc);
        std::cout << "saved " << epPath << "\n";
        logCheckpoint(epoch, epPath);
    }
}

// ================= SGD momentum 速度缓冲（CPU 训练态，逐层） =================
// 与逐层可训练参数（KERNELS_L[i]/BIAS_L[i] + FC_W/FC_b）同形。
// velocity 仅在训练期间存在，不参与 saveModel/loadModel——保持模型文件格式不变。
static K4 VEL_KERNELS_L[MAX_DEPTH];
static std::vector<float> VEL_BIAS_L[MAX_DEPTH];
// 阶段6：1×1 投影残差的 momentum 速度缓冲（仅 res==2 层非空）。
static K4 VEL_PROJ_K_L[MAX_DEPTH];
static std::vector<float> VEL_PROJ_B_L[MAX_DEPTH];
static std::vector<std::vector<float>> VEL_FC_W;
static std::vector<float> VEL_FC_b;

// 依据当前全局权重与 FC 维度，重建并清零全部 velocity 缓冲。
// momentum 是逐参数递归，初值必须为 0；在 CPU 训练开始前调用一次。
static void resetVelocity(const FullyConnected& fc) {
    for (int i = 0; i < g_net.depth; ++i) {
        VEL_KERNELS_L[i].assign(KERNELS_L[i].size(),
            std::vector<std::vector<std::vector<float>>>());
        for (size_t oc = 0; oc < KERNELS_L[i].size(); ++oc) {
            VEL_KERNELS_L[i][oc].assign(KERNELS_L[i][oc].size(),
                std::vector<std::vector<float>>());
            for (size_t ic = 0; ic < KERNELS_L[i][oc].size(); ++ic)
                VEL_KERNELS_L[i][oc][ic].assign(KERNELS_L[i][oc][ic].size(),
                    std::vector<float>(KERNELS_L[i][oc][ic][0].size(), 0.0f));
        }
        VEL_BIAS_L[i].assign(BIAS_L[i].size(), 0.0f);
    }

    // 阶段6：投影残差速度缓冲（与 PROJ_K_L/PROJ_B_L 同形；非 res==2 层清空）。
    for (int i = 0; i < g_net.depth; ++i) {
        VEL_PROJ_K_L[i].assign(PROJ_K_L[i].size(),
            std::vector<std::vector<std::vector<float>>>());
        for (size_t oc = 0; oc < PROJ_K_L[i].size(); ++oc) {
            VEL_PROJ_K_L[i][oc].assign(PROJ_K_L[i][oc].size(),
                std::vector<std::vector<float>>());
            for (size_t ic = 0; ic < PROJ_K_L[i][oc].size(); ++ic)
                VEL_PROJ_K_L[i][oc][ic].assign(PROJ_K_L[i][oc][ic].size(),
                    std::vector<float>(1, 0.0f));
        }
        VEL_PROJ_B_L[i].assign(PROJ_B_L[i].size(), 0.0f);
    }

    VEL_FC_W.assign(fc.W.size(), std::vector<float>());
    for (size_t o = 0; o < fc.W.size(); ++o)
        VEL_FC_W[o].assign(fc.W[o].size(), 0.0f);
    VEL_FC_b.assign(fc.b.size(), 0.0f);
}

// 统一 momentum 更新：v = β·v + g；Nesterov 时 step = g + β·v，否则 step = v。
// 首步 v 初值 0，故 v = g，自然避免起步偏小（无需特殊分支）。
// β=0 时 step = g，退化为朴素 SGD。
static inline void momentumStep(float& w, float& v, float g,
                                float lr, float beta, bool nesterov) {
    v = beta * v + g;
    const float step = nesterov ? (g + beta * v) : v;
    w -= lr * step;
}

// ================= 读图训练 =================
// 通过开关控制更新哪些层
int trainOne(ConvMap& input, int label, FullyConnected& fc,
    float lr_fc, float lr_conv1, float lr_conv2,
    float momentum, bool nesterov,
    bool train_conv1, bool train_conv2, bool train_fc,
    float wd,
    float& outLoss)
{
    // 逐层前向（CPU 训练不支持 BN）：Conv → [pre-act 残差(resPos=1)] → ReLU → [Pool] → [post 残差(resPos=0)]。
    // 缓存各层：cLayer=ReLU 输入（BN 输出或叠加 skip 后）、rLayer=ReLU 输出、pLayer=该层最终输出。
    std::vector<ConvMap> cLayer, rLayer, pLayer;
    cLayer.reserve(g_net.depth); rLayer.reserve(g_net.depth); pLayer.reserve(g_net.depth);
    ConvMap cur = input;
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        ConvMap c = conv(cur, KERNELS_L[i], BIAS_L[i], L.stride, L.pad);
        ConvMap h = c;                                  // ReLU 输入
        const bool haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < (int)pLayer.size());
        if (haveSrc && L.resPos == 1) {
            if (L.res == 1) addInPlace(h, pLayer[L.resFrom]);
            else {
                ConvMap proj = conv(pLayer[L.resFrom], PROJ_K_L[i], PROJ_B_L[i], 1, 0);
                addInPlace(h, proj);
            }
        }
        ConvMap r = relu(h);
        ConvMap p = L.pool ? Pool2x2(r) : r;
        if (haveSrc && L.resPos == 0) {
            if (L.res == 1) addInPlace(p, pLayer[L.resFrom]);
            else {
                ConvMap proj = conv(pLayer[L.resFrom], PROJ_K_L[i], PROJ_B_L[i], 1, 0);
                addInPlace(p, proj);
            }
        }
        cLayer.push_back(std::move(h));
        rLayer.push_back(std::move(r));
        pLayer.push_back(std::move(p));
        cur = pLayer.back();
    }
    ConvMap& p2 = pLayer.back();

    std::vector<float> flat = flatten(p2);
    std::vector<float> logits = fc.forward(flat);
    std::vector<float> prob = softmax(logits);

    outLoss = -log(prob[label] + CNN_LOSS_EPS);

    std::vector<float> dlogits = prob;
    dlogits[label] -= 1.0f;

    std::vector<std::vector<float>> dW;
    std::vector<float> db, dflat;
    FC_backward(flat, dlogits, fc.W, dW, db, dflat);

    if (train_fc) {
        for (int o = 0; o < fc.outDim; ++o) {   // H2：类别数由 FC 输出维驱动
            momentumStep(fc.b[o], VEL_FC_b[o], db[o], lr_fc, momentum, nesterov);
            for (int i = 0; i < (int)flat.size(); ++i) {
                float gW = dW[o][i];
                if (wd != 0.0f) gW += wd * fc.W[o][i];   // L2 仅权重，bias 不衰减
                momentumStep(fc.W[o][i], VEL_FC_W[o][i], gW,
                    lr_fc, momentum, nesterov);
            }
        }
    }

    // d_pool：dflat 还原为末层池化输出梯度；阶段3 改为逐层缓冲 dPool[i]。
    // dPool[i] 由「后续层主干反传」与「所有以 i 为 skip 源的残差层分流」共同累加。
    std::vector<ConvMap> dPool;
    dPool.reserve(g_net.depth);
    for (int i = 0; i < g_net.depth; ++i) {
        const LayerSpec& L = g_net.layer[i];
        const int ph = L.pool ? (L.H / 2) : L.H;
        const int pw = L.pool ? (L.W / 2) : L.W;
        dPool.emplace_back(L.outC, ph, pw);
    }
    {
        int idx = 0;
        ConvMap& last = dPool[g_net.depth - 1];
        for (int c = 0; c < last.C; ++c)
            for (int h = 0; h < last.H; ++h)
                for (int w = 0; w < last.W; ++w)
                    last.data[c][h][w] = dflat[idx++];
    }

    // 逐层反向（末层 → 首层），运算顺序与改造前逐层展开一致。
    for (int i = g_net.depth - 1; i >= 0; --i) {
        const LayerSpec& L = g_net.layer[i];
        const bool  doTrain = (i == 0) ? train_conv1 : (i == 1) ? train_conv2 : true;
        const float lr      = (i == 0) ? lr_conv1   : (i == 1) ? lr_conv2   : lr_conv2;
        const bool  haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < i);

        // resPos==0：残差加在最终输出，梯度从 dPool[i] 分流（identity 直接加；投影走反传）。
        if (haveSrc && L.resPos == 0) {
            if (L.res == 1) {
                addInPlace(dPool[L.resFrom], dPool[i]);
            } else {
                K4 dPK; std::vector<float> dPB; ConvMap dPrev;
                conv_backward(pLayer[L.resFrom], dPool[i], PROJ_K_L[i], 1, 0, dPK, dPB, dPrev);
                if (doTrain) {
                    for (int oc = 0; oc < (int)PROJ_K_L[i].size(); ++oc) {
                        momentumStep(PROJ_B_L[i][oc], VEL_PROJ_B_L[i][oc], dPB[oc],
                            lr, momentum, nesterov);
                        for (int ic = 0; ic < (int)PROJ_K_L[i][oc].size(); ++ic) {
                            float gP = dPK[oc][ic][0][0];
                            if (wd != 0.0f) gP += wd * PROJ_K_L[i][oc][ic][0][0];   // L2 仅权重
                            momentumStep(PROJ_K_L[i][oc][ic][0][0],
                                VEL_PROJ_K_L[i][oc][ic][0][0], gP,
                                lr, momentum, nesterov);
                        }
                    }
                }
                addInPlace(dPool[L.resFrom], dPrev);
            }
        }

        ConvMap d_r = L.pool ? pool2x2_backward(rLayer[i], dPool[i]) : dPool[i];
        ConvMap d_h = relu_backward(cLayer[i], d_r);

        // resPos==1：残差加在 ReLU 前，梯度从 d_h 分流。
        if (haveSrc && L.resPos == 1) {
            if (L.res == 1) {
                addInPlace(dPool[L.resFrom], d_h);
            } else {
                K4 dPK; std::vector<float> dPB; ConvMap dPrev;
                conv_backward(pLayer[L.resFrom], d_h, PROJ_K_L[i], 1, 0, dPK, dPB, dPrev);
                if (doTrain) {
                    for (int oc = 0; oc < (int)PROJ_K_L[i].size(); ++oc) {
                        momentumStep(PROJ_B_L[i][oc], VEL_PROJ_B_L[i][oc], dPB[oc],
                            lr, momentum, nesterov);
                        for (int ic = 0; ic < (int)PROJ_K_L[i][oc].size(); ++ic) {
                            float gP = dPK[oc][ic][0][0];
                            if (wd != 0.0f) gP += wd * PROJ_K_L[i][oc][ic][0][0];   // L2 仅权重
                            momentumStep(PROJ_K_L[i][oc][ic][0][0],
                                VEL_PROJ_K_L[i][oc][ic][0][0], gP,
                                lr, momentum, nesterov);
                        }
                    }
                }
                addInPlace(dPool[L.resFrom], dPrev);
            }
        }

        K4 dK;
        std::vector<float> dB;
        ConvMap dprev;
        const ConvMap& convIn = (i == 0) ? input : pLayer[i - 1];
        conv_backward(convIn, d_h, KERNELS_L[i], L.stride, L.pad, dK, dB, dprev);

        // 训练开关/学习率按层（本阶段 2 层：层0=conv1、层1=conv2）。
        if (doTrain) {
            for (int oc = 0; oc < (int)KERNELS_L[i].size(); ++oc) {
                momentumStep(BIAS_L[i][oc], VEL_BIAS_L[i][oc], dB[oc],
                    lr, momentum, nesterov);
                for (int ic = 0; ic < (int)KERNELS_L[i][oc].size(); ++ic)
                    for (int kh = 0; kh < (int)KERNELS_L[i][oc][ic].size(); ++kh)
                        for (int kw = 0; kw < (int)KERNELS_L[i][oc][ic][kh].size(); ++kw) {
                            float gK = dK[oc][ic][kh][kw];
                            if (wd != 0.0f) gK += wd * KERNELS_L[i][oc][ic][kh][kw];   // L2 仅权重
                            momentumStep(KERNELS_L[i][oc][ic][kh][kw],
                                VEL_KERNELS_L[i][oc][ic][kh][kw], gK,
                                lr, momentum, nesterov);
                        }
            }
        }
        // 主干反传：把对上一层池化输出的梯度累加进 dPool[i-1]（残差网络可能已有分流项）。
        if (i > 0) addInPlace(dPool[i - 1], dprev);
    }

    int pred = argmaxIndex(prob.data(), (int)prob.size());
    return pred;
}

// ================= GUI 模式 =================
// GUI 识别回调：接收 32×32 画布（0/255），转为 0/1 后走与训练/测试完全相同的
// 预处理与前向路径，返回预测数字并回填置信度。
static int guiPredictImpl(const unsigned char* canvas, int w, int h, float* conf) {
    ConvMap input(1, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            input.data[0][y][x] = canvas[y * w + x] ? 1.0f : 0.0f;
    preprocessImage(input, g_prep);       // 必须与训练/测试共用同一预处理
    ConvMap p2 = forwardConv(input);
    std::vector<float> prob = softmax(g_guiFc->forward(flatten(p2)));
    int pred = argmaxIndex(prob.data(), (int)prob.size());
    if (conf) *conf = prob[pred];
    return pred;
}

// GUI 模式入口：解析模型/预处理参数，加载模型后进入消息循环。
// 其余训练相关参数（--epochs/--batch/--gpu/--cpu/--data2 等）在 GUI 下被忽略。
static int runGuiMode(int argc, char** argv) {
    auto args = parseArgs(argc, argv, 1);

    // R4：GUI 独立解析参数，需与主流程（bkc 等）一致地解析保存格式开关，
    //     否则无参 GUI 的默认模型名（model.acm / model.txt）会与全局不一致。
    //     CNN2_TXTMODEL 非空且非 "0" → 文本；命令行 --txtmodel 优先。
    {
        char* envTxt = nullptr; size_t envLen = 0;
        if (_dupenv_s(&envTxt, &envLen, "CNN2_TXTMODEL") == 0 && envTxt) {
            if (envTxt[0] != '\0' && std::string(envTxt) != "0") g_saveAcm = false;
            free(envTxt);
        }
    }
    if (args.count("txtmodel")) g_saveAcm = false;   // --txtmodel：默认名切 txt

    std::string modelPath = args.count("model") ? args["model"]
        : (g_saveAcm ? std::string("model.acm") : std::string("model.txt"));
    std::string fpngPath = args.count("fpng") ? args["fpng"] : std::string();

    // D2：GUI 只支持 model/fpng/BN/bn/prep*/diag 这些键；其余命令行键被忽略，
    //     明确打印警告（替代原先的静默忽略），避免用户误以为参数已生效（如 --gpu）。
    static const char* kGuiSupported[] = {
        "model", "fpng", "BN", "bn", "txtmodel",
        "prep", "prep_target", "prep_stretch", "prep_ink", "prep_deskew_max",
        "diag", "gui"
    };
    const int kGuiSupportedN = (int)(sizeof(kGuiSupported) / sizeof(kGuiSupported[0]));
    std::vector<std::string> ignoredArgs;
    for (std::map<std::string, std::string>::const_iterator it = args.begin();
         it != args.end(); ++it) {
        bool supported = false;
        for (int s = 0; s < kGuiSupportedN; ++s)
            if (it->first == kGuiSupported[s]) { supported = true; break; }
        if (!supported) ignoredArgs.push_back(it->first);
    }
    if (!ignoredArgs.empty()) {
        std::cerr << "警告: GUI 模式忽略以下参数:";
        for (size_t s = 0; s < ignoredArgs.size(); ++s)
            std::cerr << " --" << ignoredArgs[s];
        std::cerr << "\n（GUI 恒为 CPU 单模型推理；如需这些能力请用 cnn2 <目录> test）\n";
    }

    // D1：GUI 仅支持单模型。--model 含逗号（多模型集成）时明确报错，避免把整串当单路径。
    std::vector<std::string> guiModels = splitModelList(modelPath);
    if (guiModels.size() > 1) {
        std::cerr << "GUI 模式不支持多模型集成。请使用: cnn2 <测试目录> test --model "
                  << modelPath << "\n";
        return 1;
    }
    if (guiModels.size() == 1) modelPath = guiModels[0];

    g_prep = prepConfigFromArgs(args);

    // 阶段0+1：先预览权重文件维度决定 g_net，再据此构造卷积核与全连接层
    NetSpec spec;
    std::string perr;
    if (!peekModelDims(modelPath, spec, &perr)) {
        std::cerr << "GUI: 无法读取模型维度 " << modelPath
            << "（" << perr << "）\n";
        return 1;
    }
    g_net = spec;
    // BN 与 CLI 一致性检测
    bool guiCliBn = (args.count("BN") != 0);   // 归一化后 bn 已并入 BN，无需再判 bn
    if (!checkBnCliAgreement(g_net.bn, guiCliBn, "GUI")) return 1;
    // P1：GUI 为纯 CPU 推理，放行 BN 模型（forwardConv 内按 running 统计量做 bn_infer）。
    buildKernels();
    FullyConnected fc(g_net.flat, g_net.classes);
    if (!loadModel(modelPath, fc)) {
        std::cerr << "GUI: 无法加载模型 " << modelPath << "\n";
        return 1;
    }
    // D3：与 bkc/test 路径一致，加载后校验模型形状与 g_net 自洽（否则 flatten/scatter 会越界）。
    if (!checkModelDims(modelPath, fc)) return 1;
    g_guiFc = &fc;
    return runGUI(&guiPredictImpl, 32, fpngPath.empty() ? nullptr : fpngPath.c_str());
}

// TUI 模式入口：与 gui 并列的独立模式。注入补全词典后进入全屏界面。
static int runTuiModeCli(int argc, char** argv) {
    int n = 0;
    const char* const* keys = cnn2TuiKnownArgKeys(&n);
    return runTuiMode(argc, argv, keys, n);
}

// ============================================================
// 有限差分梯度校验（--gradcheck）
// ------------------------------------------------------------
// 目的：用中心差分数值梯度对拍「解析梯度」，重点覆盖阶段3 残差层的
//       梯度分流，以及 BN 的 γ/β 反向。只新增只读校验路径：
//       * 自建小网络（c=4）+ 小批量（默认 8）+ 固定种子，不依赖模型文件；
//       * 解析梯度复用既有反向原语（conv_backward/relu_backward/
//         pool2x2_backward/FC_backward），残差分流顺序与 trainOne 完全一致；
//       * BN 走训练态 batch 统计（有偏方差），反向公式与 gpu_kernels.cu 的
//         kbn_bwd 完全一致（CPU 训练本不支持 BN，故此处内置同公式实现用于
//         对拍；若改 GPU BN 反向，需同步本公式）；
//       * 未启用 --gradcheck 时，本段代码完全不被调用，默认行为逐位不变。
// ============================================================

// 参数类别
enum GCCat { GC_K = 0, GC_B, GC_G, GC_BETA, GC_PW, GC_PB, GC_FCW, GC_FCB };
static const char* gcCatName(int c) {
    switch (c) {
        case GC_K:    return "conv_w";
        case GC_B:    return "conv_b";
        case GC_G:    return "bn_gamma";
        case GC_BETA: return "bn_beta";
        case GC_PW:   return "proj_w";
        case GC_PB:   return "proj_b";
        case GC_FCW:  return "fc_w";
        case GC_FCB:  return "fc_b";
        default:      return "unknown";
    }
}

// 指向某个可训练标量的引用（layer=-1 表示 FC 段）
struct GCParamRef { float* ptr; int cat; int layer; };

struct GCNet {
    NetSpec spec;
    K4 K[MAX_DEPTH];
    std::vector<float> B[MAX_DEPTH];
    std::vector<float> G[MAX_DEPTH], Bt[MAX_DEPTH];   // BN γ / β
    K4 PK[MAX_DEPTH];                                  // 阶段6：1×1 投影权重 [outC][srcC][1][1]
    std::vector<float> PB[MAX_DEPTH];                  // 阶段6：投影偏置
    std::vector<std::vector<float>> W;                 // FC 权重 [o][i]
    std::vector<float> fb;                             // FC 偏置
};

struct GCAct {
    std::vector<std::vector<ConvMap>> cPre;   // [层][样本] 卷积输出（BN 前）
    std::vector<std::vector<ConvMap>> bnOut;  // [层][样本] BN 输出（ReLU 前）；非 BN 同 cPre
    std::vector<std::vector<ConvMap>> r;      // [层][样本] ReLU 输出
    std::vector<std::vector<ConvMap>> p;      // [层][样本] 该层最终输出（池化+残差）
    std::vector<std::vector<float>> bnMean;   // [层][C]
    std::vector<std::vector<float>> bnInv;    // [层][C] = 1/sqrt(var+eps)
};

// 固定种子初始化小网络权重（与 buildKernels 的 time() 种子无关，保证可复现）
static void gcInitNet(GCNet& net, const NetSpec& spec, unsigned seed) {
    net.spec = spec;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> convDist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> fcDist(-0.05f, 0.05f);
    int inC = 1;
    for (int i = 0; i < spec.depth; ++i) {
        const LayerSpec& L = spec.layer[i];
        net.K[i].assign(L.outC, std::vector<std::vector<std::vector<float>>>(
            inC, std::vector<std::vector<float>>(
                L.kH, std::vector<float>(L.kW, 0.0f))));
        for (int oc = 0; oc < L.outC; ++oc)
            for (int ic = 0; ic < inC; ++ic)
                for (int kh = 0; kh < L.kH; ++kh)
                    for (int kw = 0; kw < L.kW; ++kw)
                        net.K[i][oc][ic][kh][kw] = convDist(rng);
        net.B[i].assign(L.outC, 0.0f);
        net.G[i].assign(L.outC, 1.0f);
        net.Bt[i].assign(L.outC, 0.0f);
        // 阶段6：投影残差权重初始化（仅 res==2）
        if (L.res == 2 && L.resFrom >= 0 && L.resFrom < i) {
            const int SC = spec.layer[L.resFrom].outC;
            net.PK[i].assign(L.outC, std::vector<std::vector<std::vector<float>>>(
                SC, std::vector<std::vector<float>>(
                    1, std::vector<float>(1, 0.0f))));
            for (int oc = 0; oc < L.outC; ++oc)
                for (int ic = 0; ic < SC; ++ic)
                    net.PK[i][oc][ic][0][0] = convDist(rng);
            net.PB[i].assign(L.outC, 0.0f);
        } else {
            net.PK[i].clear();
            net.PB[i].clear();
        }
        inC = L.outC;
    }
    net.W.assign(spec.classes, std::vector<float>(spec.flat, 0.0f));
    for (int o = 0; o < spec.classes; ++o)
        for (int k = 0; k < spec.flat; ++k)
            net.W[o][k] = fcDist(rng);
    net.fb.assign(spec.classes, 0.0f);
}

// 前向：act 非空时缓存反向所需激活；返回各样本 logits。
// 残差语义与 forwardConv/trainOne 一致：p = pool(relu(bn(conv(x)))) + p[resFrom]。
static void gcForward(const GCNet& net, const std::vector<ConvMap>& x,
                      GCAct* act, std::vector<std::vector<float>>& logits) {
    const NetSpec& s = net.spec;
    const int N = (int)x.size();
    GCAct local;
    GCAct& a = act ? *act : local;
    a.cPre.assign(s.depth, {});    a.bnOut.assign(s.depth, {});
    a.r.assign(s.depth, {});       a.p.assign(s.depth, {});
    a.bnMean.assign(s.depth, {});  a.bnInv.assign(s.depth, {});

    std::vector<ConvMap> cur = x;
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        std::vector<ConvMap> cpre(N);
        for (int n = 0; n < N; ++n)
            cpre[n] = conv(cur[n], net.K[i], net.B[i], L.stride, L.pad);

        std::vector<ConvMap> bn;
        std::vector<float> mean, inv;
        if (s.bn) {
            const int C = L.outC, H = cpre[0].H, W = cpre[0].W;
            const float cnt = (float)(N * H * W);
            mean.assign(C, 0.0f); inv.assign(C, 0.0f);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            mean[c] += cpre[n].data[c][h][w];
            for (int c = 0; c < C; ++c) mean[c] /= cnt;
            std::vector<float> var(C, 0.0f);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const float d = cpre[n].data[c][h][w] - mean[c];
                            var[c] += d * d;
                        }
            for (int c = 0; c < C; ++c) inv[c] = 1.0f / std::sqrt(var[c] / cnt + CNN_BN_EPS);
            bn.resize(N);
            for (int n = 0; n < N; ++n) {
                bn[n] = ConvMap(C, H, W);
                for (int c = 0; c < C; ++c) {
                    const float g = net.G[i][c], b = net.Bt[i][c];
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            bn[n].data[c][h][w] =
                            g * (cpre[n].data[c][h][w] - mean[c]) * inv[c] + b;
                }
            }
        } else {
            bn = cpre;
        }

        std::vector<ConvMap> rr(N), pp(N), bnIn(N);
        const bool haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < i);
        for (int n = 0; n < N; ++n) {
            bnIn[n] = bn[n];                     // ReLU 输入（BN 输出，resPos=1 时再叠加 skip）
            if (haveSrc && L.resPos == 1) {
                if (L.res == 1) addInPlace(bnIn[n], a.p[L.resFrom][n]);
                else {
                    ConvMap proj = conv(a.p[L.resFrom][n], net.PK[i], net.PB[i], 1, 0);
                    addInPlace(bnIn[n], proj);
                }
            }
            rr[n] = relu(bnIn[n]);
            pp[n] = L.pool ? Pool2x2(rr[n]) : rr[n];
            if (haveSrc && L.resPos == 0) {
                if (L.res == 1) addInPlace(pp[n], a.p[L.resFrom][n]);
                else {
                    ConvMap proj = conv(a.p[L.resFrom][n], net.PK[i], net.PB[i], 1, 0);
                    addInPlace(pp[n], proj);
                }
            }
        }

        if (act) {
            a.cPre[i] = std::move(cpre); a.bnOut[i] = std::move(bnIn);
            a.r[i] = std::move(rr);
            a.bnMean[i] = std::move(mean); a.bnInv[i] = std::move(inv);
        }
        a.p[i] = std::move(pp);
        cur = a.p[i];
    }

    logits.assign(N, std::vector<float>(s.classes, 0.0f));
    for (int n = 0; n < N; ++n) {
        std::vector<float> flat = flatten(cur[n]);
        for (int o = 0; o < s.classes; ++o) {
            float v = net.fb[o];
            for (int k = 0; k < s.flat; ++k) v += net.W[o][k] * flat[k];
            logits[n][o] = v;
        }
    }
}

// ---- 双精度前向：仅用于数值侧，消除 float 前向在中心差分中的舍入噪声 ----
// 解析侧仍走既有 float 原语（conv/conv_backward 等），二者对拍即可暴露反向 bug。
struct ConvMapD {
    int C, H, W;
    std::vector<std::vector<std::vector<double>>> d;
    ConvMapD() : C(0), H(0), W(0) {}
    ConvMapD(int c, int h, int w) : C(c), H(h), W(w),
        d(c, std::vector<std::vector<double>>(h, std::vector<double>(w, 0.0))) {}
};

static ConvMapD gcConvD(const ConvMapD& in, const K4& K,
                        const std::vector<float>& B, int stride, int pad) {
    const int IC = in.C;
    const int KH = (int)K[0][0].size(), KW = (int)K[0][0][0].size();
    const int OC = (int)K.size();
    const int outH = (in.H + 2 * pad - KH) / stride + 1;
    const int outW = (in.W + 2 * pad - KW) / stride + 1;
    ConvMapD out(OC, outH, outW);
    for (int oc = 0; oc < OC; ++oc)
        for (int oh = 0; oh < outH; ++oh)
            for (int ow = 0; ow < outW; ++ow) {
                double sum = (double)B[oc];
                for (int ic = 0; ic < IC; ++ic)
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            const int ih = oh * stride + kh - pad;
                            const int iw = ow * stride + kw - pad;
                            if (ih < 0 || ih >= in.H || iw < 0 || iw >= in.W) continue;
                            sum += in.d[ic][ih][iw] * (double)K[oc][ic][kh][kw];
                        }
                out.d[oc][oh][ow] = sum;
            }
    return out;
}

// 双精度前向：输出各样本 double softmax 概率；key 非空时输出激活键。
static void gcForwardD(const GCNet& net, const std::vector<ConvMap>& x,
                       std::vector<std::vector<double>>& probs, std::vector<int>* key) {
    const NetSpec& s = net.spec;
    const int N = (int)x.size();
    if (key) key->clear();
    std::vector<ConvMapD> cur(N);
    for (int n = 0; n < N; ++n) {
        cur[n] = ConvMapD(1, s.inH, s.inW);
        for (int h = 0; h < s.inH; ++h)
            for (int w = 0; w < s.inW; ++w)
                cur[n].d[0][h][w] = (double)x[n].data[0][h][w];
    }
    std::vector<std::vector<ConvMapD>> pAll(s.depth);
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        std::vector<ConvMapD> cpre(N);
        for (int n = 0; n < N; ++n)
            cpre[n] = gcConvD(cur[n], net.K[i], net.B[i], L.stride, L.pad);

        std::vector<ConvMapD> bn;
        if (s.bn) {
            const int C = L.outC, H = cpre[0].H, W = cpre[0].W;
            const double cnt = (double)(N * H * W);
            std::vector<double> mean(C, 0.0), var(C, 0.0), inv(C, 0.0);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) mean[c] += cpre[n].d[c][h][w];
            for (int c = 0; c < C; ++c) mean[c] /= cnt;
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const double dd = cpre[n].d[c][h][w] - mean[c];
                            var[c] += dd * dd;
                        }
            for (int c = 0; c < C; ++c) inv[c] = 1.0 / std::sqrt(var[c] / cnt + 1e-5);
            bn.resize(N);
            for (int n = 0; n < N; ++n) {
                bn[n] = ConvMapD(C, H, W);
                for (int c = 0; c < C; ++c) {
                    const double gg = (double)net.G[i][c], bb = (double)net.Bt[i][c];
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            bn[n].d[c][h][w] =
                            gg * (cpre[n].d[c][h][w] - mean[c]) * inv[c] + bb;
                }
            }
        } else {
            bn = cpre;
        }

        std::vector<ConvMapD> rr(N), pp(N), bnIn(N);
        const bool haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < i);
        for (int n = 0; n < N; ++n) {
            bnIn[n] = bn[n];                     // ReLU 输入（resPos=1 时叠加 skip）
            if (haveSrc && L.resPos == 1) {
                if (L.res == 1) {
                    for (int c = 0; c < bnIn[n].C; ++c)
                        for (int h = 0; h < bnIn[n].H; ++h)
                            for (int w = 0; w < bnIn[n].W; ++w)
                                bnIn[n].d[c][h][w] += pAll[L.resFrom][n].d[c][h][w];
                } else {
                    ConvMapD proj = gcConvD(pAll[L.resFrom][n], net.PK[i], net.PB[i], 1, 0);
                    for (int c = 0; c < bnIn[n].C; ++c)
                        for (int h = 0; h < bnIn[n].H; ++h)
                            for (int w = 0; w < bnIn[n].W; ++w)
                                bnIn[n].d[c][h][w] += proj.d[c][h][w];
                }
            }
            rr[n] = ConvMapD(bnIn[n].C, bnIn[n].H, bnIn[n].W);
            for (int c = 0; c < bnIn[n].C; ++c)
                for (int h = 0; h < bnIn[n].H; ++h)
                    for (int w = 0; w < bnIn[n].W; ++w) {
                        const double v = bnIn[n].d[c][h][w];
                        rr[n].d[c][h][w] = v > 0.0 ? v : 0.0;
                    }
            if (L.pool) {
                const int oh2 = rr[n].H / 2, ow2 = rr[n].W / 2;
                pp[n] = ConvMapD(rr[n].C, oh2, ow2);
                for (int c = 0; c < rr[n].C; ++c)
                    for (int oh = 0; oh < oh2; ++oh)
                        for (int ow = 0; ow < ow2; ++ow) {
                            double m = rr[n].d[c][oh * 2][ow * 2];
                            double v1 = rr[n].d[c][oh * 2][ow * 2 + 1];
                            double v2 = rr[n].d[c][oh * 2 + 1][ow * 2];
                            double v3 = rr[n].d[c][oh * 2 + 1][ow * 2 + 1];
                            if (v1 > m) m = v1; if (v2 > m) m = v2; if (v3 > m) m = v3;
                            pp[n].d[c][oh][ow] = m;
                        }
            } else {
                pp[n] = rr[n];
            }
            if (haveSrc && L.resPos == 0) {
                if (L.res == 1) {
                    for (int c = 0; c < pp[n].C; ++c)
                        for (int h = 0; h < pp[n].H; ++h)
                            for (int w = 0; w < pp[n].W; ++w)
                                pp[n].d[c][h][w] += pAll[L.resFrom][n].d[c][h][w];
                } else {
                    ConvMapD proj = gcConvD(pAll[L.resFrom][n], net.PK[i], net.PB[i], 1, 0);
                    for (int c = 0; c < pp[n].C; ++c)
                        for (int h = 0; h < pp[n].H; ++h)
                            for (int w = 0; w < pp[n].W; ++w)
                                pp[n].d[c][h][w] += proj.d[c][h][w];
                }
            }
        }

        if (key) {
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < bnIn[n].C; ++c)
                    for (int h = 0; h < bnIn[n].H; ++h)
                        for (int w = 0; w < bnIn[n].W; ++w) {
                            const double v = bnIn[n].d[c][h][w];
                            key->push_back(v > 0.0 ? 1 : (v < 0.0 ? -1 : 0));
                        }
            if (L.pool) {
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < rr[n].C; ++c)
                        for (int oh = 0; oh + 1 < rr[n].H; oh += 2)
                            for (int ow = 0; ow + 1 < rr[n].W; ow += 2) {
                                int idx = 0;
                                double m = rr[n].d[c][oh][ow];
                                if (rr[n].d[c][oh][ow + 1] > m) { m = rr[n].d[c][oh][ow + 1]; idx = 1; }
                                if (rr[n].d[c][oh + 1][ow] > m) { m = rr[n].d[c][oh + 1][ow]; idx = 2; }
                                if (rr[n].d[c][oh + 1][ow + 1] > m) { idx = 3; }
                                key->push_back(idx);
                            }
            }
        }
        pAll[i] = std::move(pp);
        cur = pAll[i];
    }

    probs.assign(N, std::vector<double>(s.classes, 0.0));
    for (int n = 0; n < N; ++n) {
        std::vector<double> flat;
        const ConvMapD& last = pAll[s.depth - 1][n];
        for (int c = 0; c < last.C; ++c)
            for (int h = 0; h < last.H; ++h)
                for (int w = 0; w < last.W; ++w)
                    flat.push_back(last.d[c][h][w]);
        std::vector<double> z(s.classes, 0.0);
        for (int o = 0; o < s.classes; ++o) {
            double v = (double)net.fb[o];
            for (int k = 0; k < s.flat; ++k) v += (double)net.W[o][k] * flat[k];
            z[o] = v;
        }
        double mx = z[0];
        for (int o = 1; o < s.classes; ++o) if (z[o] > mx) mx = z[o];
        double sum = 0.0;
        for (int o = 0; o < s.classes; ++o) { probs[n][o] = std::exp(z[o] - mx); sum += probs[n][o]; }
        for (int o = 0; o < s.classes; ++o) probs[n][o] /= sum;
    }
}

// 双精度小批量平均交叉熵（数值侧）
static double gcLossD(const std::vector<std::vector<double>>& probs,
                      const std::vector<int>& y) {
    const int N = (int)probs.size();
    double L = 0.0;
    for (int n = 0; n < N; ++n)
        L += -std::log(probs[n][y[n]] + 1e-300);
    return L / (double)N;
}

// 解析反向：复用既有反向原语；残差分流顺序与 trainOne 一致；BN 公式对齐 kbn_bwd。
// 输出 gana，顺序与 gcBuildRefs 完全一致。
static void gcAnalyticGrad(const GCNet& net, const std::vector<ConvMap>& x,
                           const std::vector<int>& y,
                           const GCAct& a, std::vector<float>& gana) {
    const NetSpec& s = net.spec;
    const int N = (int)x.size();

    // ---- FC 段（损失对 batch 取平均：dlogits 已含 1/N）----
    std::vector<std::vector<float>> flatAll(N), dflat(N);
    std::vector<std::vector<float>> gW(s.classes, std::vector<float>(s.flat, 0.0f));
    std::vector<float> gfb(s.classes, 0.0f);
    for (int n = 0; n < N; ++n) flatAll[n] = flatten(a.p[s.depth - 1][n]);

    for (int n = 0; n < N; ++n) {
        std::vector<float> logits(s.classes, 0.0f);
        for (int o = 0; o < s.classes; ++o) {
            float v = net.fb[o];
            for (int k = 0; k < s.flat; ++k) v += net.W[o][k] * flatAll[n][k];
            logits[o] = v;
        }
        std::vector<float> dlogits = softmax(logits);
        dlogits[y[n]] -= 1.0f;
        dflat[n].assign(s.flat, 0.0f);
        for (int o = 0; o < s.classes; ++o) {
            const float dl = dlogits[o] / (float)N;
            gfb[o] += dl;
            for (int k = 0; k < s.flat; ++k) {
                gW[o][k] += dl * flatAll[n][k];
                dflat[n][k] += dl * net.W[o][k];
            }
        }
    }

    // ---- 逐层反向（末层 → 首层）----
    std::vector<std::vector<ConvMap>> dPool(s.depth, std::vector<ConvMap>(N));
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        const int ph = L.pool ? (L.H / 2) : L.H;
        const int pw = L.pool ? (L.W / 2) : L.W;
        for (int n = 0; n < N; ++n) dPool[i][n] = ConvMap(L.outC, ph, pw);
    }
    for (int n = 0; n < N; ++n) {
        ConvMap& last = dPool[s.depth - 1][n];
        int idx = 0;
        for (int c = 0; c < last.C; ++c)
            for (int h = 0; h < last.H; ++h)
                for (int w = 0; w < last.W; ++w)
                    last.data[c][h][w] = dflat[n][idx++];
    }

    K4 gK[MAX_DEPTH];
    std::vector<float> gB[MAX_DEPTH], gG[MAX_DEPTH], gBt[MAX_DEPTH];
    K4 gPK[MAX_DEPTH];                        // 阶段6：投影权重梯度
    std::vector<float> gPB[MAX_DEPTH];        // 阶段6：投影偏置梯度
    for (int i = 0; i < s.depth; ++i) {
        const LayerSpec& L = s.layer[i];
        const int inC = (i == 0) ? 1 : s.layer[i - 1].outC;
        gK[i].assign(L.outC, std::vector<std::vector<std::vector<float>>>(
            inC, std::vector<std::vector<float>>(
                L.kH, std::vector<float>(L.kW, 0.0f))));
        gB[i].assign(L.outC, 0.0f);
        gG[i].assign(L.outC, 0.0f);
        gBt[i].assign(L.outC, 0.0f);
        if (L.res == 2 && L.resFrom >= 0 && L.resFrom < i) {
            const int SC = s.layer[L.resFrom].outC;
            gPK[i].assign(L.outC, std::vector<std::vector<std::vector<float>>>(
                SC, std::vector<std::vector<float>>(
                    1, std::vector<float>(1, 0.0f))));
            gPB[i].assign(L.outC, 0.0f);
        }
    }

    for (int i = s.depth - 1; i >= 0; --i) {
        const LayerSpec& L = s.layer[i];
        const bool haveSrc = (L.res != 0 && L.resFrom >= 0 && L.resFrom < i);

        // resPos==0：残差加在最终输出，梯度从 dPool[i] 分流（顺序与 trainOne 一致）
        if (haveSrc && L.resPos == 0) {
            if (L.res == 1) {
                for (int n = 0; n < N; ++n) addInPlace(dPool[L.resFrom][n], dPool[i][n]);
            } else {
                for (int n = 0; n < N; ++n) {
                    K4 dPK; std::vector<float> dPB; ConvMap dprev;
                    conv_backward(a.p[L.resFrom][n], dPool[i][n], net.PK[i], 1, 0, dPK, dPB, dprev);
                    for (int oc = 0; oc < L.outC; ++oc) {
                        gPB[i][oc] += dPB[oc];
                        for (int ic = 0; ic < (int)dPK[oc].size(); ++ic)
                            gPK[i][oc][ic][0][0] += dPK[oc][ic][0][0];
                    }
                    addInPlace(dPool[L.resFrom][n], dprev);
                }
            }
        }

        // 过 Pool / ReLU（a.bnOut 为 ReLU 输入：resPos=1 时已含 skip）
        std::vector<ConvMap> dBn(N);
        for (int n = 0; n < N; ++n) {
            ConvMap d_r = L.pool ? pool2x2_backward(a.r[i][n], dPool[i][n]) : dPool[i][n];
            dBn[n] = relu_backward(a.bnOut[i][n], d_r);
        }

        // resPos==1：残差加在 ReLU 前，梯度从 dBn 分流
        if (haveSrc && L.resPos == 1) {
            if (L.res == 1) {
                for (int n = 0; n < N; ++n) addInPlace(dPool[L.resFrom][n], dBn[n]);
            } else {
                for (int n = 0; n < N; ++n) {
                    K4 dPK; std::vector<float> dPB; ConvMap dprev;
                    conv_backward(a.p[L.resFrom][n], dBn[n], net.PK[i], 1, 0, dPK, dPB, dprev);
                    for (int oc = 0; oc < L.outC; ++oc) {
                        gPB[i][oc] += dPB[oc];
                        for (int ic = 0; ic < (int)dPK[oc].size(); ++ic)
                            gPK[i][oc][ic][0][0] += dPK[oc][ic][0][0];
                    }
                    addInPlace(dPool[L.resFrom][n], dprev);
                }
            }
        }

        // 过 BN（跨 batch 耦合）或恒等
        std::vector<ConvMap> dCpre(N);
        if (s.bn) {
            const int C = L.outC, H = a.cPre[i][0].H, W = a.cPre[i][0].W;
            const float cnt = (float)(N * H * W);
            std::vector<float> S1(C, 0.0f), S2(C, 0.0f);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const float dy = dBn[n].data[c][h][w];
                            const float xhat =
                                (a.cPre[i][n].data[c][h][w] - a.bnMean[i][c]) * a.bnInv[i][c];
                            S1[c] += dy; S2[c] += dy * xhat;
                            gG[i][c] += dy * xhat; gBt[i][c] += dy;
                        }
            for (int n = 0; n < N; ++n) {
                dCpre[n] = ConvMap(C, H, W);
                for (int c = 0; c < C; ++c) {
                    const float gy = net.G[i][c], inv = a.bnInv[i][c];
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const float dy = dBn[n].data[c][h][w];
                            const float xhat =
                                (a.cPre[i][n].data[c][h][w] - a.bnMean[i][c]) * inv;
                            dCpre[n].data[c][h][w] =
                                gy * inv * (1.0f / cnt) * (cnt * dy - S1[c] - xhat * S2[c]);
                        }
                }
            }
        } else {
            dCpre = dBn;
        }

        // 过 Conv（逐样本累加），并把对上一层池化输出的梯度加回 dPool[i-1]
        for (int n = 0; n < N; ++n) {
            const ConvMap& convIn = (i == 0) ? x[n] : a.p[i - 1][n];
            K4 dKn; std::vector<float> dBn2; ConvMap dprev;
            conv_backward(convIn, dCpre[n], net.K[i], L.stride, L.pad, dKn, dBn2, dprev);
            for (int oc = 0; oc < L.outC; ++oc)
                for (int ic = 0; ic < (int)dKn[oc].size(); ++ic)
                    for (int kh = 0; kh < L.kH; ++kh)
                        for (int kw = 0; kw < L.kW; ++kw)
                            gK[i][oc][ic][kh][kw] += dKn[oc][ic][kh][kw];
            for (int oc = 0; oc < L.outC; ++oc) gB[i][oc] += dBn2[oc];
            if (i > 0) addInPlace(dPool[i - 1][n], dprev);
        }
    }

    // ---- 打包 gana（与 gcBuildRefs 同序）----
    gana.clear();
    for (int i = 0; i < s.depth; ++i) {
        for (int oc = 0; oc < s.layer[i].outC; ++oc)
            for (int ic = 0; ic < (int)gK[i][oc].size(); ++ic)
                for (int kh = 0; kh < s.layer[i].kH; ++kh)
                    for (int kw = 0; kw < s.layer[i].kW; ++kw)
                        gana.push_back(gK[i][oc][ic][kh][kw]);
        for (int oc = 0; oc < s.layer[i].outC; ++oc) gana.push_back(gB[i][oc]);
        if (s.bn) {
            for (int c = 0; c < s.layer[i].outC; ++c) gana.push_back(gG[i][c]);
            for (int c = 0; c < s.layer[i].outC; ++c) gana.push_back(gBt[i][c]);
        }
        // 阶段6：投影段（仅 res==2；与 gcBuildRefs 同序）
        if (s.layer[i].res == 2 && !gPK[i].empty()) {
            for (int oc = 0; oc < s.layer[i].outC; ++oc)
                for (int ic = 0; ic < (int)gPK[i][oc].size(); ++ic)
                    gana.push_back(gPK[i][oc][ic][0][0]);
            for (int oc = 0; oc < s.layer[i].outC; ++oc) gana.push_back(gPB[i][oc]);
        }
    }
    for (int o = 0; o < s.classes; ++o)
        for (int k = 0; k < s.flat; ++k) gana.push_back(gW[o][k]);
    for (int o = 0; o < s.classes; ++o) gana.push_back(gfb[o]);
}

// 按打包顺序建立「标量指针」引用表
static void gcBuildRefs(GCNet& net, std::vector<GCParamRef>& refs) {
    refs.clear();
    for (int i = 0; i < net.spec.depth; ++i) {
        for (auto& oc : net.K[i])
            for (auto& ic : oc)
                for (auto& row : ic)
                    for (float& v : row) refs.push_back({ &v, GC_K, i });
        for (float& v : net.B[i]) refs.push_back({ &v, GC_B, i });
        if (net.spec.bn) {
            for (float& v : net.G[i])  refs.push_back({ &v, GC_G, i });
            for (float& v : net.Bt[i]) refs.push_back({ &v, GC_BETA, i });
        }
        if (net.spec.layer[i].res == 2) {
            for (auto& oc : net.PK[i])
                for (auto& ic : oc)
                    for (auto& row : ic)
                        for (float& v : row) refs.push_back({ &v, GC_PW, i });
            for (float& v : net.PB[i]) refs.push_back({ &v, GC_PB, i });
        }
    }
    for (auto& row : net.W) for (float& v : row) refs.push_back({ &v, GC_FCW, -1 });
    for (float& v : net.fb) refs.push_back({ &v, GC_FCB, -1 });
}

// 判定某层是否与残差相关（是残差层本身，或被别的层当作 skip 源）
static bool gcLayerResidual(const NetSpec& s, int li) {
    if (s.layer[li].res != 0) return true;
    for (int j = 0; j < s.depth; ++j)
        if (s.layer[j].res != 0 && s.layer[j].resFrom == li) return true;
    return false;
}

// 运行单个网络类型的梯度校验；返回该类型是否 PASS，并回填整体最大 relErr。
static bool gcRunCase(const std::string& name, const NetSpec& spec,
                      unsigned weightSeed, int maxSamples, float eps,
                      float thr, double& outMax) {
    GCNet net;
    gcInitNet(net, spec, weightSeed);

    std::vector<GCParamRef> refs;
    gcBuildRefs(net, refs);

    // ---- 构造小批量（固定种子；折点由下方激活一致性判据逐参数处理）----
    const int N = 8;
    std::vector<ConvMap> x(N);
    std::vector<int> y(N);
    std::mt19937 brng(0x51ed2701u ^ weightSeed);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    for (int n = 0; n < N; ++n) {
        x[n] = ConvMap(1, spec.inH, spec.inW);
        for (int h = 0; h < spec.inH; ++h)
            for (int w = 0; w < spec.inW; ++w)
                x[n].data[0][h][w] = (ud(brng) < 0.3f) ? 1.0f : 0.0f;
    }
    for (int n = 0; n < N; ++n) y[n] = n % spec.classes;

    // ---- 解析梯度 ----
    GCAct act;
    std::vector<std::vector<float>> logits;
    gcForward(net, x, &act, logits);
    std::vector<float> gana;
    gcAnalyticGrad(net, x, y, act, gana);

    // ---- 抽样：按 (类别,层) 分组；残差相关层重点覆盖（尽量全测）----
    struct GCGroup { int cat; int layer; std::vector<int> idx; };
    std::vector<GCGroup> groups;
    for (int p = 0; p < (int)refs.size(); ++p) {
        int gi = -1;
        for (int q = 0; q < (int)groups.size(); ++q)
            if (groups[q].cat == refs[p].cat && groups[q].layer == refs[p].layer) { gi = q; break; }
        if (gi < 0) { groups.push_back({ refs[p].cat, refs[p].layer, {} }); gi = (int)groups.size() - 1; }
        groups[gi].idx.push_back(p);
    }

    std::mt19937 srng(0xabcdef01u ^ weightSeed);
    std::vector<int> sampled;
    for (auto& g : groups) {
        std::vector<int> idx = g.idx;
        std::shuffle(idx.begin(), idx.end(), srng);
        int cap = maxSamples;
        const bool resLayer = (g.layer >= 0) && gcLayerResidual(spec, g.layer);
        if (resLayer || (int)idx.size() <= 256) cap = (int)idx.size();
        if (cap > (int)idx.size()) cap = (int)idx.size();
        for (int q = 0; q < cap; ++q) sampled.push_back(idx[q]);
    }

    // ---- 数值梯度（中心差分）并对拍 ----
    // 逐参数做「激活一致性」校验：若 w±eps 处 ReLU 符号/池化 argmax 与区间一致，
    // 则该区间内函数局部线性，中心差分精确；否则逐步减半 eps 重试；仍冲突则跳过。
    std::map<int, double> catMax;
    int catCount[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int kinkSkipped = 0;
    int nearZero = 0;
    double maxAll = 0.0;      // 仅统计「有意义梯度」的相对误差
    double maxAbsErr = 0.0;   // 近零梯度的绝对误差
    bool absFail = false;
    for (int p : sampled) {
        float* w = refs[p].ptr;
        const float save = *w;
        double gnum = 0.0;
        double ee = (double)eps;
        bool smooth = false;
        for (int attempt = 0; attempt < 40; ++attempt) {
            std::vector<std::vector<double>> lgp, lgm;
            std::vector<int> kp, km;
            *w = save + (float)ee;
            gcForwardD(net, x, lgp, &kp);
            const double Lp = gcLossD(lgp, y);
            *w = save - (float)ee;
            gcForwardD(net, x, lgm, &km);
            const double Lm = gcLossD(lgm, y);
            if (kp == km) {
                gnum = (Lp - Lm) / (2.0 * ee);
                smooth = true;
                break;
            }
            ee *= 0.5;
            if (ee < 1e-6) break;
        }
        *w = save;
        if (!smooth) { ++kinkSkipped; continue; }

        const double ganaP = (double)gana[p];
        const double denom = std::fabs(gnum) + std::fabs(ganaP);
        const double absFloor = 1e-5;   // 解析侧为 float，绝对噪声 ~1e-7
        if (denom < absFloor) {
            // 近零梯度（典型：BN 之前的卷积偏置，解析上恒为 0；float 解析侧
            // 仍有 ~1e-7 舍入残余）。相对误差无意义，改用绝对判据，不计入 maxRelErr。
            ++nearZero;
            const double absErr = std::fabs(gnum - ganaP);
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (absErr > absFloor) absFail = true;
            continue;
        }
        const double rel = std::fabs(gnum - ganaP) / denom;
        auto it = catMax.find(refs[p].cat);
        if (it == catMax.end()) catMax[refs[p].cat] = rel;
        else if (rel > it->second) it->second = rel;
        catCount[refs[p].cat]++;
        if (rel > maxAll) maxAll = rel;
    }
    outMax = maxAll;

    bool pass = (kinkSkipped < (int)sampled.size()) &&
                (maxAll < (double)thr) && !absFail;
    std::cout << "[GRADCHECK] " << name
              << " sampled=" << sampled.size() << " kinkSkipped=" << kinkSkipped
              << " nearZero=" << nearZero << " maxAbsErr(nearZero)=" << maxAbsErr
              << " eps=" << eps << " thr=" << thr << "\n";
    for (int c = 0; c < 8; ++c) {
        if (catCount[c] == 0) continue;
        std::cout << "[GRADCHECK] " << name << " " << gcCatName(c)
                  << " maxRelErr=" << catMax[c] << " n=" << catCount[c] << "\n";
    }
    std::cout << "[GRADCHECK] " << name << " maxRelErr=" << maxAll
              << " " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

static int runGradCheck(std::map<std::string, std::string> args) {
    int maxSamples = 48;
    float eps = 1e-3f;
    try {
        if (args.count("gc_samples")) maxSamples = std::stoi(args["gc_samples"]);
        if (args.count("gc_eps"))     eps = std::stof(args["gc_eps"]);
    } catch (...) { std::cerr << "--gradcheck: gc_samples/gc_eps 非法\n"; return 1; }
    if (maxSamples < 1) maxSamples = 1;

    std::cout << "[GRADCHECK] 开始：小网络(c=4)+小批量(8)+固定种子，中心差分对拍解析梯度\n";

    struct Case { std::string name; NetSpec spec; unsigned seed; bool bn; };
    std::vector<Case> cases;

    // (a) depth=2 非 BN
    cases.push_back({ "depth2",      makeDefaultNetSpec(32, 32, 10, false, 2, 4, 4), 1001u, false });
    // (b) depth=2 BN（训练态 batch 统计）
    cases.push_back({ "depth2bn",    makeDefaultNetSpec(32, 32, 10, true, 2, 4, 4),  2002u, true });
    // (c) depth=3 残差（identity skip，末层指向 source 层）
    cases.push_back({ "depth3res",   makeResNetSpec(32, 32, 10, false, 3, 4),       3003u, false });
    // (d) depth=3 1×1 投影残差（res=2，纯 CPU 权重对拍 proj_w/proj_b）
    cases.push_back({ "projres",     makeProjResNetSpec(32, 32, 10, false, 3, 4),   4004u, false });
    // (e) depth=3 pre-activation 残差（resPos=1，残差加在 BN 后/ReLU 前）
    cases.push_back({ "preactres",   makePreactResNetSpec(32, 32, 10, false, 3, 4), 5005u, false });

    int failed = 0;
    for (auto& c : cases) {
        std::string verr;
        if (!validateNetSpec(c.spec, &verr)) {
            std::cerr << "[GRADCHECK] " << c.name << " 网络规格非法: " << verr << "\n";
            ++failed;
            continue;
        }
        // BN 的数值条件数较差，阈值适当放宽（见任务说明），非 BN/残差严格 1e-2
        const float thr = c.bn ? 3e-2f : 1e-2f;
        double maxAll = 0.0;
        if (!gcRunCase(c.name, c.spec, c.seed, maxSamples, eps, thr, maxAll)) ++failed;
    }

    if (failed == 0) {
        std::cout << "[GRADCHECK] SUMMARY PASS\n";
        return 0;
    }
    std::cout << "[GRADCHECK] SUMMARY FAIL (" << failed << " 个网络类型未通过)\n";
    return 1;
}

// ================= --version：打印版本后正常退出 =================
// 不加载模型/数据、不训练、不初始化 CUDA；打印程序版本与模型格式版本后返回 0。
// 版本常量集中在 version.h，便于统一维护。
static int runVersion() {
    std::cout << "CNN2 程序版本: " << CNN2_VERSION << "\n";
    std::cout << "模型格式版本: ver=" << CNN2_MODEL_VER
              << "（对应 #SIG 行中的 ver 字段）\n";
    std::cout << "构建日期: " << CNN2_BUILD_DATE << " " << CNN2_BUILD_TIME << "\n";
    std::cout << "编译器: MSVC / C++17（占位，详见 README 构建章节）\n";
    std::cout << "CUDA: 运行时信息占位（--version 不初始化 CUDA）\n";
    return 0;
}

// ================= T6：read 命令（读取/查看模型文件） =================
// 轻量入口（不建核、不训练、不推理、不修改全局权重）：
//   探测格式 + 预览形状（peekModelDimsTxt / decodeAcmHeader）并打印模型配置摘要。
// 用法：cnn2 read <文件>   或   cnn2 read --model <文件>；成功退出 0，失败 1。
static int runReadModel(int argc, char** argv) {
    // ---- 解析模型路径：优先位置参数（argv[2] 非 -- 开头），否则 --model ----
    std::string path;
    if (argc >= 3) {
        const std::string a2 = argv[2];
        const bool isOpt = (a2.size() > 2 && a2[0] == '-' && a2[1] == '-');
        if (!isOpt) path = a2;
    }
    if (path.empty()) {
        std::map<std::string, std::string> args = parseArgs(argc, argv, 1);
        if (args.count("model")) path = args["model"];
    }
    if (path.empty()) {
        std::cerr << "用法: cnn2 read <模型文件>\n";
        std::cerr << "      cnn2 read --model <模型文件>\n";
        return 1;
    }
    if (!fileExists(path)) {
        std::cerr << "read: 找不到文件: " << path << "\n";
        return 1;
    }

    // ---- 文件大小（字节） ----
    long long fileSize = -1;
    {
        std::ifstream fsz(path, std::ios::binary);
        if (fsz) { fsz.seekg(0, std::ios::end); fileSize = (long long)fsz.tellg(); }
    }

    const ModelFormat fmt = probeModelFormat(path);
    if (fmt == ModelFormat::Unknown) {
        std::cerr << "read: " << path
                  << " 后缀为 .acm 但魔数不符（文件损坏或并非 acm 格式）\n";
        return 1;
    }
    const bool isAcm = (fmt == ModelFormat::Acm);

    NetSpec spec;
    std::string sig;
    std::string err;
    int headerVersion = 0, formatVer = 1, precision = 0, flags = 0;

    if (isAcm) {
        // 只读 1024B 固定头（不读数据区、不建核）。
        std::ifstream fin(path, std::ios::binary);
        if (!fin) { std::cerr << "read: 打不开文件: " << path << "\n"; return 1; }
        std::vector<uint8_t> hdr((size_t)ACM_HEADER_SIZE, 0);
        fin.read(reinterpret_cast<char*>(hdr.data()), (std::streamsize)ACM_HEADER_SIZE);
        if ((size_t)fin.gcount() < (size_t)ACM_HEADER_SIZE) {
            std::cerr << "read: acm 头部不完整（不足 1024 字节）: " << path << "\n";
            return 1;
        }
        AcmHeader h;
        if (!decodeAcmHeader(hdr, h, spec, sig, &err)) {
            std::cerr << "read: " << err << " (" << path << ")\n";
            return 1;
        }
        headerVersion = h.headerVersion;
        formatVer     = h.formatVersion;
        precision     = h.precision;
        flags         = (int)h.flags;
    } else {
        // txt：预览形状；另读首行 #SIG 供 ver 显示（BOM/CR 容错，与 peek 一致）。
        if (!peekModelDimsTxt(path, spec, &err)) {
            std::cerr << "read: " << err << " (" << path << ")\n";
            return 1;
        }
        std::ifstream fin(path);
        if (fin) {
            std::string first;
            std::getline(fin, first);
            if (first.size() >= 3 &&
                (unsigned char)first[0] == 0xEF &&
                (unsigned char)first[1] == 0xBB &&
                (unsigned char)first[2] == 0xBF) first.erase(0, 3);
            if (!first.empty() && first.back() == '\r') first.pop_back();
            size_t b = first.find_first_not_of(" \t");
            if (b != std::string::npos && first[b] == '#') sig = first.substr(b);
        }
        formatVer = acmSigVersion(sig);
    }

    const int ver = acmSigVersion(sig);   // #SIG 的 ver（acm/txt 统一解析）

    // ---- 输出模型配置摘要 ----
    std::cout << "模型文件: " << path << "\n";
    if (isAcm) {
        std::cout << "格式: acm(二进制)"
                  << "  头版本=" << headerVersion
                  << "  ver=" << formatVer
                  << "  精度=" << (precision == ACM_PREC_FP32
                        ? std::string("fp32") : std::to_string((int)precision))
                  << "  flags=0x" << std::hex << flags << std::dec
                  << "  文件大小=" << fileSize << " 字节\n";
    } else {
        std::cout << "格式: txt(文本)"
                  << "  ver=" << formatVer
                  << "  文件大小=" << fileSize << " 字节\n";
    }
    std::cout << "#SIG 等价: BN=" << (spec.bn ? "ON" : "OFF")
              << " depth=" << spec.depth
              << " ver=" << ver << "\n";
    std::cout << "网络: 输入 " << spec.inH << "x" << spec.inW
              << "  类别 " << spec.classes
              << "  深度 " << spec.depth
              << "  参数量 total=" << spec.total << "\n";
    std::cout << "逐层:\n";
    for (int i = 0; i < spec.depth; ++i) {
        const LayerSpec& L = spec.layer[i];
        std::cout << "  L" << (i + 1)
                  << ": K " << L.outC << "x" << L.inC << "x" << L.kH << "x" << L.kW
                  << "  B " << L.outC
                  << "  P(" << L.stride << "," << L.pad << "," << L.pool << ")";
        if (L.res != 0)
            std::cout << "  R" << L.res << "(from=" << L.resFrom << ",pos=" << L.resPos << ")";
        std::cout << "\n";
    }
    std::cout << "FC: " << spec.classes << "x" << spec.flat << "\n";
    return 0;
}

// ================= --help：打印帮助后正常退出 =================
// 优先级最高（见 main 最前分流）：一旦出现 --help，忽略其余所有参数，输出帮助并退出 0。
//   * 若指定了模式（如 `cnn2 bkc --help`）→ 输出该模式的专属帮助；
//   * 否则（如 `cnn2 --help`）            → 输出软件总说明。
// 模式判定：扫描 argv 中首个「模式词」（gen/bkc/train/test/infer/gui/tui/read/
//           checkdata/gradcheck/version，裸词或 --xxx 形式均可）。

// 将单个 token 映射为内部模式名；非模式词返回空串。
static std::string helpModeFromToken(const std::string& tok) {
    std::string k = tok;
    if (k.size() > 2 && k[0] == '-' && k[1] == '-')      k = k.substr(2);  // --xxx
    else if (k.size() > 1 && k[0] == '-')                k = k.substr(1);  // -x
    if (k == "train") return "bkc";     // train ≡ bkc
    if (k == "gen" || k == "bkc" || k == "test" || k == "infer" || k == "gui" ||
        k == "tui" || k == "read" || k == "checkdata" || k == "gradcheck" || k == "version")
        return k;
    return std::string();
}

// 预处理参数段（训练/测试/推理/单图/GUI 共用），供各模式帮助复用。
static void printPrepHelp() {
    std::cout << "预处理参数（训练/测试/推理共用）:\n";
    std::cout << "  --prep crop,resize,center   启用的预处理步骤（逗号分隔）/ none 关闭全部\n";
    std::cout << "                              可选: crop,deskew,resize,center,thick\n";
    std::cout << "  --prep_target N             resize 目标框最长边（16-30，默认 28）\n";
    std::cout << "  --prep_stretch              拉伸到 target×target（否则保持长宽比）\n";
    std::cout << "  --prep_ink X                目标墨迹比（thick 用，默认 0.20，0=关闭）\n";
    std::cout << "  --prep_deskew_max D         deskew 角度限幅（度，默认 15）\n";
    std::cout << "  --diag                      打印预处理前后统计（墨迹比/质心/包围盒）\n";
}

// 打印软件总说明（无模式时）。
static void printGeneralHelp() {
    std::cout << "CNN2 —— 手写数字卷积神经网络 训练/推理工具\n";
    std::cout << "程序版本: " << CNN2_VERSION << "   模型格式版本: ver=" << CNN2_MODEL_VER
              << "   构建: " << CNN2_BUILD_DATE << " " << CNN2_BUILD_TIME << "\n\n";
    std::cout << "用法:\n";
    std::cout << "  cnn2 <模式> [路径] [选项]      新式子命令写法\n";
    std::cout << "  cnn2 <路径> <模式> [选项]      旧式写法（模式紧随路径）\n\n";
    std::cout << "模式:\n";
    std::cout << "  gen        生成/初始化模型（网络结构见下方专属参数）\n";
    std::cout << "  bkc        训练模型（同义子命令: train）\n";
    std::cout << "  test       批量测试/评估（支持多模型集成与 TTA）\n";
    std::cout << "  infer      单图推理（默认模式）\n";
    std::cout << "  gui        图形界面（Win32 手写识别）\n";
    std::cout << "  tui        终端 TUI 指令台（自动补全，便于试跑命令）\n";
    std::cout << "  read       读取/查看模型文件\n";
    std::cout << "  checkdata  数据目录校验\n";
    std::cout << "  gradcheck  有限差分梯度校验\n";
    std::cout << "  version    打印程序版本与模型格式版本\n\n";
    std::cout << "通用/全局参数:\n";
    std::cout << "  --gpu / --cpu               设备选择（默认按模式；test 默认 CPU）\n";
    std::cout << "  --model <file>              模型文件（默认 model.txt；test 可逗号分隔多模型）\n";
    std::cout << "  --batch N                   批大小（默认 128）\n";
    std::cout << "  --threads N                 GPU 数据加载并行线程数（默认 8；1=单线程；0=auto）\n";
    std::cout << "  --BN (--bn)                 启用 BatchNorm（生成/训练/推理 BN 模型须一致）\n";
    std::cout << "  --depth N                   网络深度（默认 2；gen/bkc 新建生效）\n";
    std::cout << "  --wd F                      L2 weight decay（默认 0.0）\n";
    std::cout << "  --epsave [目录]             每轮保存权重（仅 bkc）\n";
    std::cout << "  --epsn N                    每轮存档间隔（默认 1）\n";
    std::cout << "  --txtmodel                  以纯文本格式保存模型（读取恒自动识别格式；与 .acm 后缀同用会报错）\n";
    std::cout << "  --help                      显示帮助（本说明；`cnn2 <模式> --help` 看模式专属）\n\n";
    printPrepHelp();
    std::cout << "\n常用示例:\n";
    std::cout << "  cnn2 gen model.txt --depth 4 --BN\n";
    std::cout << "  cnn2 train data/ --epochs 10 --gpu --log train.csv --val val/\n";
    std::cout << "  cnn2 test test/ --model a.txt,b.txt --fuse soft --tta\n";
    std::cout << "  cnn2 infer img.txt --model model.txt\n";
    std::cout << "  cnn2 checkdata data/ --strict-data\n\n";
    std::cout << "提示: cnn2 <模式> --help 查看该模式专属参数（如 cnn2 bkc --help）。\n";
}

// 打印指定模式的专属帮助。
static void printModeHelp(const std::string& mode) {
    if (mode == "gen") {
        std::cout << "cnn2 gen —— 生成/初始化模型\n";
        std::cout << "用法: cnn2 gen [路径] [--model model.txt] [网络结构参数]\n\n";
        std::cout << "  --model <file>     输出/保存的模型文件（默认 model.txt；--out 等价）\n";
        std::cout << "  --depth N          网络深度（默认 2；depth>2 使用默认 schedule 并写 #SIG ver=2）\n";
        std::cout << "  --c1 N             Conv1 输出通道数（默认 32）\n";
        std::cout << "  --c2 N             Conv2 输出通道数（默认 64）\n";
        std::cout << "  --BN (--bn)        生成带 BatchNorm 的模型\n";
        std::cout << "  --P                签名行写完整环境（GPU名+计算机名+时间+BN+depth）\n";
        std::cout << "  --res              生成 identity 残差网（末层 skip 指向 stem）\n";
        std::cout << "  --res2             生成 1×1 投影残差网（Wproj 通道对齐）\n";
        std::cout << "  --preact           生成 pre-activation 残差网（残差加在 BN 后/ReLU 前）\n";
        std::cout << "  --no_conv1/--no_conv2/--no_fc   初始化时冻结对应层\n";
        std::cout << "  --txtmodel         以纯文本格式保存模型（格式按后缀：.txt 文本 / .acm 二进制 / 无后缀视开关）\n";
    } else if (mode == "bkc") {
        std::cout << "cnn2 bkc —— 训练模型（同义: cnn2 train <目录>）\n";
        std::cout << "用法: cnn2 bkc <训练目录> [选项]\n\n";
        std::cout << "训练超参:\n";
        std::cout << "  --epochs N         训练轮数（默认 10）\n";
        std::cout << "  --lr_fc X          FC 学习率（默认 0.0001）\n";
        std::cout << "  --lr_conv1 X       Conv1 学习率（默认 0.0001）\n";
        std::cout << "  --lr_conv2 X       Conv2 学习率（默认 0.0005）\n";
        std::cout << "  --momentum B       SGD momentum 系数 β（默认 0.9；0 关闭为朴素 SGD）\n";
        std::cout << "  --nesterov         启用 Nesterov look-ahead 动量\n";
        std::cout << "  --onecycle         启用 One-Cycle 学习率调度\n";
        std::cout << "  --lr_warmup N      warmup 轮数（默认 auto=max(1,epochs/5)）\n";
        std::cout << "  --lr_min F         最小 lr 因子（默认 0.01）\n";
        std::cout << "  --lr_peak F        峰值 lr 因子（默认 1.0）\n";
        std::cout << "  --batch N          批大小（默认 128）\n";
        std::cout << "  --wd F             L2 weight decay（默认 0.0；仅卷积核/投影权/FC权）\n";
        std::cout << "  --epsave [目录]    每轮保存权重（默认 EPTSET\\<模型名>\\；仅 bkc）\n";
        std::cout << "  --epsn N           每轮存档间隔（默认 1；仅 --epsave 开启时生效）\n";
        std::cout << "  --val <目录>       独立验证集（每 epoch 评估；配合 --log 存最优 <model>.best）\n";
        std::cout << "  --log <file>       追加训练日志（每 epoch 一行 CSV）\n";
        std::cout << "  --time             输出训练耗时\n";
        std::cout << "  --gpu / --cpu      设备选择\n";
        std::cout << "  --threads N        GPU 数据加载并行线程数（默认 8）\n";
        std::cout << "  --seed N           随机种子\n";
        std::cout << "  --no_conv1/--no_conv2/--no_fc   冻结对应层（不训练）\n";
        std::cout << "\n混合训练:\n";
        std::cout << "  --data2 <目录>     第二套数据目录（混合训练）\n";
        std::cout << "  --mix <mode>       混合策略: balanced(默认,域均衡) | full(全量投喂)\n";
        std::cout << "  --full             等价于 --mix full\n";
        std::cout << "\n数据增强（默认关闭）:\n";
        std::cout << "  --aug / --aug_k K / --aug_shift X / --aug_rotate X / --aug_scale X\n";
        std::cout << "  --aug_elastic A / --aug_elastic_sigma S / --aug_thick P\n";
        printPrepHelp();
    } else if (mode == "test") {
        std::cout << "cnn2 test —— 批量测试/评估\n";
        std::cout << "用法: cnn2 test <测试目录> [选项]（test 默认 CPU）\n\n";
        std::cout << "  --model a.txt,b.txt,...   多模型集成（逗号分隔；逐类 softmax 平均后 argmax）\n";
        std::cout << "  --fuse <mode>             集成融合: soft(默认)|logit|geom|vote|gate|wbest\n";
        std::cout << "  --weights w1,w2,...       每模型权重（默认全 1）\n";
        std::cout << "  --fuse_margin M           gate 模式独醒者反选阈值（默认 0.0）\n";
        std::cout << "  --fuse_temp T             概率温度缩放（默认 1.0）\n";
        std::cout << "  --tta                     开启测试时增强（log 概率空间平均）\n";
        std::cout << "  --tta_k K                 TTA 变换数量（默认 5，最多 11）\n";
        std::cout << "  --outerror                输出判错样本清单（屏幕 + outerror.txt）\n";
        std::cout << "  --batch N                 批大小\n";
        std::cout << "  --gpu / --cpu             设备选择\n";
        printPrepHelp();
    } else if (mode == "infer") {
        std::cout << "cnn2 infer —— 单图推理（默认模式）\n";
        std::cout << "用法: cnn2 <测试图> [--model model.txt]\n\n";
        std::cout << "  --model <file>     模型文件（默认 model.txt）\n";
        std::cout << "  --cpu / --gpu      设备选择\n";
        printPrepHelp();
    } else if (mode == "gui") {
        std::cout << "cnn2 gui —— 图形界面\n";
        std::cout << "用法: cnn2 gui [--model 路径] [--fpng 图片txt路径] [--prep ...]\n\n";
        std::cout << "  --model <file>     默认加载 model.txt\n";
        std::cout << "  --fpng <file>      启动时载入画布的 32×32 txt 图片\n";
        printPrepHelp();
    } else if (mode == "tui") {
        // 与 TUI 内建 help 共用同一份文案（tui/TuiHelpText.h）。
        std::cout << cnn2::tui::kTuiHelpText;
    } else if (mode == "read") {
        std::cout << "cnn2 read —— 读取/查看模型文件\n";
        std::cout << "用法: cnn2 read <模型文件>\n\n";
        std::cout << "  读取并显示模型的 #SIG 元信息与层/形状结构（不训练、不推理）。\n";
    } else if (mode == "checkdata") {
        std::cout << "cnn2 checkdata —— 数据目录校验\n";
        std::cout << "用法: cnn2 checkdata <目录> [--strict-data]（或 cnn2 <目录> --checkdata）\n\n";
        std::cout << "  扫描并报告文件数 / 每类计数 / 非法文件（非 32×32、非 0/1、空、首字符非数字）。\n";
        std::cout << "  --strict-data   发现非法文件时报错并非零退出\n";
    } else if (mode == "gradcheck") {
        std::cout << "cnn2 gradcheck —— 有限差分梯度校验\n";
        std::cout << "用法: cnn2 gradcheck [--gc_samples K] [--gc_eps E]（或 cnn2 --gradcheck）\n\n";
        std::cout << "  自建小网络对拍残差/BN/conv/fc；PASS 退出 0，FAIL 非零退出。\n";
        std::cout << "  --gc_samples K   采样点数\n";
        std::cout << "  --gc_eps E       有限差分步长\n";
    } else if (mode == "version") {
        std::cout << "cnn2 version —— 打印版本\n";
        std::cout << "用法: cnn2 version（或 cnn2 --version）\n\n";
        std::cout << "  打印程序版本、模型格式版本、构建日期/时间后退出（不加载数据/不训练）。\n";
    } else {
        printGeneralHelp();
    }
}

// 帮助入口：判定模式并输出对应帮助，始终返回 0。
static int runHelp(int argc, char** argv) {
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help") continue;
        const std::string m = helpModeFromToken(a);
        if (!m.empty()) { mode = m; break; }
    }
    printModeHelp(mode);
    return 0;
}

// ================= main =================
int main(int argc, char** argv) {
    // TUI 子进程：stdout 被重定向到管道时 CRT 默认全缓冲，会导致输出攒块。
    // 仅在 TUI 通过环境变量明确要求时改用无缓冲，普通终端/重定向行为完全不变。
    // 说明：
    //  1) MSVC UCRT 实测在管道上 _IOLBF 不生效（仍整块缓冲），只有 _IONBF 能实时输出，
    //     因此这里用无缓冲；TUI 子进程输出量小，额外系统调用开销可忽略。
    //  2) UCRT 的 setvbuf 在 buffer=nullptr 且 size=0 时会触发 invalid parameter
    //     （进程直接 fastfail），所以必须显式给出非 0 缓冲区大小。
    {
        char* linebufEnv = nullptr; size_t linebufLen = 0;
        if (_dupenv_s(&linebufEnv, &linebufLen, "CNN2_STDOUT_LINEBUF") == 0 && linebufEnv) {
            if (linebufEnv[0] != '\0') std::setvbuf(stdout, nullptr, _IONBF, 4096);
            free(linebufEnv);
        }
    }

    // --help 分流：优先级最高（高于 --cnn2/--version/--gui/--gradcheck/--checkdata 等）。
    // 一旦出现 --help，忽略其余所有参数，输出帮助（模式说明或总说明）后退出 0。
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--help") return runHelp(argc, argv);

    // 彩蛋分流（--cnn2）：命中即在最前打印 ASCII art，然后【继续正常流程】（不退出、不影响后续）。
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cnn2") {
            const char* art[] = {
                R"ART(    ___   ______ ______ _   __ _   __ ___ )ART",
                R"ART(   /   | /_  __// ____// | / // | / /|__ \)ART",
                R"ART(  / /| |  / /  / /    /  |/ //  |/ / __/ /)ART",
                R"ART( / ___ | / /  / /___ / /|  // /|  / / __/ )ART",
                R"ART(/_/  |_|/_/   \____//_/ |_//_/ |_/ /____/ )ART",
            };
            for (const char* line : art) {
                std::cout << line << '\n';
            }
            break;   // 只打印一次
        }
    }

    // 版本模式分流：优先于一切（不加载数据/模型、不训练），打印后退出 0。
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--version") return runVersion();

    // GUI / TUI 模式分流：按命令行中出现顺序，先出现者优先（优先于普通参数解析）
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gui") return runGuiMode(argc, argv);
        if (a == "--tui") return runTuiModeCli(argc, argv);
    }

    // 梯度校验模式分流：自建小网络，不依赖其余参数/模型文件
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gradcheck")
            return runGradCheck(parseArgs(argc, argv, 1));

    // 数据校验模式分流（阶段4.4）：cnn2 <目录> --checkdata [--strict-data]
    // 只扫描目录、打印报告后退出，不训练、不推理、不加载模型。
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--checkdata") {
            if (argc < 2 || (argv[1][0] == '-' && argv[1][1] == '-')) {
                std::cerr << "用法: cnn2 <目录> --checkdata [--strict-data]\n";
                return 1;
            }
            bool strict = false;
            for (int j = 1; j < argc; ++j)
                if (std::string(argv[j]) == "--strict-data") strict = true;
            int nIllegal = checkDataDir(std::string(argv[1]), strict);
            return (strict && nIllegal > 0) ? 1 : 0;
        }
    }

    // ---------- 阶段B：子命令（B3，F1 修正） ----------
    // 新式：cnn2 <子命令> <路径> [选项]；旧式：cnn2 <路径> <模式> [选项] 继续有效。
    // 判别（F1）：仅当 argv[1] 恰为子命令词【且 argv[2] 不是模式词（gen/bkc/test）】时，
    //             才按「新式子命令」解析（其后首个非选项 token 为路径）；否则沿用旧式。
    //   例：cnn2 train bkc → 旧式 path=train, mode=bkc（恢复正确行为）；
    //       cnn2 train <dir> → 新式子命令 train（path=<dir>, mode=bkc）；
    //       cnn2 test <dir>  → 新式子命令 test；
    //       cnn2 test test   → 旧式 path=test, mode=test。
    const std::string arg1 = (argc >= 2) ? std::string(argv[1]) : std::string();
    const std::string arg2 = (argc >= 3) ? std::string(argv[2]) : std::string();
    const bool arg2IsMode = (arg2 == "gen" || arg2 == "bkc" || arg2 == "test");
    std::string substMode;            // 子命令映射后的内部模式（空串=单图推理）
    bool        substIsSubcmd = false;
    if (!arg2IsMode) {
        if (arg1 == "train")      { substMode = "bkc";  substIsSubcmd = true; }
        else if (arg1 == "test")  { substMode = "test"; substIsSubcmd = true; }
        else if (arg1 == "gen")   { substMode = "gen";  substIsSubcmd = true; }
        else if (arg1 == "bkc")   { substMode = "bkc";  substIsSubcmd = true; }
        else if (arg1 == "infer") { substMode = "";     substIsSubcmd = true; }  // 单图推理
    }

    // 子命令形式的工具模式（等价于对应 --flag）
    if (arg1 == "version")   return runVersion();
    if (arg1 == "read")      return runReadModel(argc, argv);
    if (arg1 == "gui")       return runGuiMode(argc, argv);
    if (arg1 == "tui")       return runTuiModeCli(argc, argv);
    if (arg1 == "gradcheck") return runGradCheck(parseArgs(argc, argv, 1));
    if (arg1 == "checkdata") {
        if (argc < 3) { std::cerr << "用法: cnn2 checkdata <目录> [--strict-data]\n"; return 1; }
        bool strict = false;
        for (int j = 1; j < argc; ++j) {
            std::string a = argv[j];
            if (a == "--strict-data" || a == "--strict_data") strict = true;
        }
        int nIllegal = checkDataDir(std::string(argv[2]), strict);
        return (strict && nIllegal > 0) ? 1 : 0;
    }

    if (argc < 2) {
        std::cerr << "用法:\n";
        std::cerr << "  cnn2 <路径> gen [--model model.txt]\n";
        std::cerr << "  cnn2 <训练目录> bkc [--epochs N] [--lr_fc X] [--lr_conv1 X] [--lr_conv2 X]\n";
        std::cerr << "                       [--momentum B] [--nesterov]\n";
        std::cerr << "                       [--no_conv1] [--no_conv2] [--no_fc]\n";
        std::cerr << "                       [--batch N] [--gpu] [--cpu] [--seed N] [--time] [--threads N]\n";
        std::cerr << "  --momentum B                SGD momentum 系数 β（默认 0.9 开启；0 关闭退化为朴素 SGD）\n";
        std::cerr << "  --nesterov                  启用 Nesterov look-ahead 动量（默认关）\n";
        std::cerr << "  --onecycle                  启用 One-Cycle 学习率调度（默认关；关闭时固定 LR 逐位一致）\n";
        std::cerr << "  --lr_warmup N               One-Cycle warmup 轮数（默认 auto = max(1, epochs/5)）\n";
        std::cerr << "  --lr_min F                  One-Cycle 最小 lr 因子（默认 0.01，最小 lr = base*F）\n";
        std::cerr << "  --lr_peak F                 One-Cycle 峰值 lr 因子（默认 1.0，峰值 lr = base*F）\n";
        std::cerr << "  --time                      输出训练耗时（每个 epoch 与总时间，CPU/GPU 均支持）\n";
        std::cerr << "  --log <file>                追加训练日志（默认关）：每 epoch 一行 CSV\n";
        std::cerr << "                              epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec；\n";
        std::cerr << "                              每次保存模型追加一行 CHECKPOINT 元数据\n";
        std::cerr << "  --threads N                 GPU 数据加载（读图+预处理+增强）并行线程数（默认 8；1=单线程 legacy；0=auto）\n";
        std::cerr << "  --val <目录>                独立验证集（默认关）：每 epoch 末评估并打印 val acc；\n";
        std::cerr << "                              配合 --log 时日志追加 val_acc 列，并把最优模型存到 <model>.best；\n";
        std::cerr << "                              与 test 同源预处理/推理，不做增强；不传时行为完全不变\n";
        std::cerr << "  --wd F                      L2 weight decay（默认 0.0）：g += wd*param，仅卷积核/投影权/FC权\n";
        std::cerr << "  --epsave [目录]             每轮保存权重（默认 EPTSET\\<模型名>\\，相对 CWD；仅 bkc 生效）\n";
        std::cerr << "  --epsn N                    每轮存档间隔（默认 1=每轮；仅 --epsave 开启时生效）\n";
        std::cerr << "  --checkdata                 数据校验模式：cnn2 <目录> --checkdata 扫描并报告\n";
        std::cerr << "                              文件数 / 每类计数 / 非法文件（非 32x32、非 0/1、空、首字符非数字）\n";
        std::cerr << "  --strict-data               校验发现非法文件时报错并非零退出（配合 --checkdata 或 bkc）\n";
        std::cerr << "  --cnn2                      显示彩蛋（不影响运行）\n";
    std::cerr << "  --P                         签名行写完整环境（GPU 名+计算机名+时间+BN+depth）；\n";
    std::cerr << "                              不加 --P 时头部仍强制写，仅含 BN:ON|OFF 与 depth=2\n";
    std::cerr << "  --BN (--bn)                 启用 BatchNorm：gen 可生成/保存带 BN 模型；\n";
    std::cerr << "                              bkc 用 --gpu 真 BN 训练；test/单图/GUI 可推理 BN 模型（test --gpu 用 running）\n";
    std::cerr << "                              加载 BN:ON 模型必须同样传 --BN；非 BN 模型不可传 --BN\n";
        std::cerr << "  --c1 N                      Conv1 输出通道数（默认 32；gen/bkc 生效；加载模型时以文件为准）\n";
        std::cerr << "  --c2 N                      Conv2 输出通道数（默认 64；gen/bkc 生效；加载模型时以文件为准）\n";
        std::cerr << "  --depth N                   网络深度 N（默认 2；gen/bkc 新建生效）。depth=2 与旧行为一致；\n";
        std::cerr << "                              depth>2 使用默认 schedule：每层 Conv(k=3,pad=1,stride=1) 保持尺寸、\n";
        std::cerr << "                              pool=2 每层减半（32→16→8→…），通道 layer0=--c1 其余=--c2，\n";
        std::cerr << "                              并在模型文件写显式 P<i> 与 #SIG ver=2\n";
        std::cerr << "  --res                       生成 identity 残差网（阶段3；末层 skip 指向 stem）\n";
        std::cerr << "  --res2                      生成 1×1 投影残差网（阶段6；末层 skip 经 Wproj 通道对齐）\n";
        std::cerr << "  --preact                    生成 pre-activation 残差网（阶段6；残差加在 BN 后/ReLU 前）\n";
        std::cerr << "\n混合训练（阶段D）:\n";
        std::cerr << "  --data2 <目录>              第二套数据目录（与主目录混合训练）\n";
        std::cerr << "  --mix <mode>                混合策略: balanced(默认,域均衡分层) | full(全量投喂)\n";
        std::cerr << "  --full                      等价于 --mix full（两域全部样本，不做下采样/均衡）\n";
        std::cerr << "\n数据增强（阶段E，默认关闭）:\n";
        std::cerr << "  --aug                        开启数据增强\n";
        std::cerr << "  --aug_k K                    离线扩增副本数（GPU路径用，默认2）\n";
        std::cerr << "  --aug_shift X                随机平移幅度 ±X px（默认3）\n";
        std::cerr << "  --aug_rotate X               随机旋转幅度 ±X 度（默认12）\n";
        std::cerr << "  --aug_scale X                缩放抖动半幅（默认0.1 => 0.9~1.1）\n";
        std::cerr << "  --aug_elastic A              弹性形变强度 α（默认0关闭，建议从4起）\n";
        std::cerr << "  --aug_elastic_sigma S        弹性位移场高斯 σ（默认4.5）\n";
        std::cerr << "  --aug_thick P                笔画粗细增强触发概率 %（默认50，0关闭）\n";
        std::cerr << "  cnn2 <测试目录> test [--model a.txt,b.txt,...] [--gpu] [--cpu] [--batch N]\n";
        std::cerr << "                              （test 默认 CPU；--gpu 启用批量 GPU 推理）\n";
        std::cerr << "  --outerror                  测试时输出判错样本清单（屏幕 + outerror.txt）\n";
        std::cerr << "  --model 可用逗号分隔多个模型做集成：对每个样本各模型 softmax\n";
        std::cerr << "          概率逐类平均后取 argmax（仅 test 生效；单模型行为不变）\n";
        std::cerr << "  --fuse <mode>               集成融合: soft(默认)|logit|geom|vote|gate|wbest\n";
        std::cerr << "  --weights w1,w2,...         每模型权重（逗号分隔，默认全 1）\n";
        std::cerr << "  --fuse_margin M             gate 模式独醒者反选阈值（默认 0.0）\n";
        std::cerr << "  --fuse_temp T               概率温度缩放（默认 1.0，p^(1/T) 后归一化）\n";
        std::cerr << "  --tta                       开启测试时增强（仅 test，log 概率空间平均）\n";
        std::cerr << "  --tta_k K                   TTA 变换数量（默认 5，最多 11）\n";
        std::cerr << "  cnn2 <测试图> [--model model.txt]\n";
    std::cerr << "  cnn2 --gui [--model 路径] [--fpng 图片txt路径] [--prep ...]\n";
    std::cerr << "                              （Win32 手写识别界面，默认加载 model.txt；\n";
    std::cerr << "                                --fpng 指定启动时载入画布的 32×32 txt 图片）\n";
    std::cerr << "  cnn2 --tui\n";
    std::cerr << "                              （终端 TUI 指令台：输入 CNN2 指令并回车执行，\n";
    std::cerr << "                                带 Tab 自动补全与 ↑↓ 历史）\n";
    std::cerr << "  cnn2 --gradcheck [--gc_samples K] [--gc_eps E]\n";
    std::cerr << "                              （有限差分梯度校验：自建小网络对拍残差/BN/conv/fc；\n";
    std::cerr << "                                PASS 退出 0，FAIL 非零退出）\n";
    std::cerr << "  cnn2 --version\n";
    std::cerr << "                              （打印程序版本与模型格式版本后退出，不加载数据/不训练）\n";
        std::cerr << "\n预处理参数（训练/测试/推理共用）:\n";
        std::cerr << "  --prep crop,resize,center   启用的预处理步骤（逗号分隔）/ none 关闭全部\n";
        std::cerr << "                              可选步骤: crop,deskew,resize,center,thick\n";
        std::cerr << "  --prep_target N             resize 目标框最长边（16-30，默认 28）\n";
        std::cerr << "  --prep_stretch              拉伸到 target×target（否则保持长宽比）\n";
        std::cerr << "  --prep_ink X                目标墨迹比（thick 用，默认 0.20，0=关闭）\n";
        std::cerr << "  --prep_deskew_max D         deskew 角度限幅（度，默认 15）\n";
        std::cerr << "  --diag                      打印预处理前后统计（墨迹比/质心/包围盒/二值断言）\n";
        return 1;
    }

    // ---------- 读取参数 ----------
    // parseStart：选项解析起点（默认跳过 argv[1]=路径 与 argv[2]=模式）。
    std::string path = argv[1];
    std::string mode = (argc >= 3) ? std::string(argv[2]) : "";
    int parseStart = 3;

    if (substIsSubcmd) {
        // 新式：argv[1]=子命令（已映射为 substMode）；其后首个非选项 token 为路径。
        // F2：本分支仅在 argv[2] 非模式词时进入（见上方 F1 判别），属正确的「新式顺序」，
        //     原生子命令写法不再打印任何「顺序颠倒」警告。
        path.clear();
        parseStart = 2;
        if (argc >= 3) {
            std::string a2 = argv[2];
            bool isOpt = (a2.size() > 2 && a2[0] == '-' && a2[1] == '-');
            if (!isOpt) { path = a2; parseStart = 3; }
        }
        mode = substMode;
    }
    else if (path.size() > 2 && path[0] == '-' && path[1] == '-') {
        // argv[1] 本身是一个选项（而非路径），例如拼错的 `cnn2 --gup`：
        // 整体按选项解析以触发 B2 未知参数警告；无路径/模式，交由默认（单图）分支处理。
        path.clear();
        mode.clear();
        parseStart = 1;
    }

    // ---------- 解析参数 ----------
    auto args = parseArgs(argc, argv, parseStart);

    // 默认值
    int   epochs = 10;
    float lr_fc = 0.0001f;
    float lr_conv1 = 0.0001f;
    float lr_conv2 = 0.0005f;
    float momentum = 0.9f;             // SGD momentum 系数 β（默认开启，0=关闭）
    bool  nesterov = false;            // 是否启用 Nesterov（默认关）
    bool  train_conv1 = true;
    bool  train_conv2 = true;
    bool  train_fc = true;
    std::string modelPath;             // T7：未显式 --model 时按格式取默认（acm/txt），见下方 g_saveAcm 确定后
    int   batch = 128;
    bool  forceGpu = false;
    bool  forceCpu = false;
    bool  timeIt = false;              // --time：输出训练耗时（默认关）
    std::string logPath;               // --log：训练日志文件（空=关闭，默认行为不变）
    std::string valDir;                // --val：独立验证集目录（空=关闭，默认行为不变）
    float wdArg = 0.0f;                // --wd：L2 weight decay 系数（默认 0.0=关闭，逐位不变）
    std::string epSaveArg;             // --epsave：目录参数（空/"1"=裸 flag，用默认 EPTSET\<模型名>\）
    bool  strictData = false;          // --strict-data：数据校验发现非法文件即退出
    int   nThreads = 8;                // --threads：数据加载线程数（默认 8；0=auto）
    bool  outError = false;            // --outerror：test 时输出判错样本清单（屏幕+outerror.txt）
    std::string data2;                 // 阶段D：第二套数据目录（空=关闭，保持旧行为）
    std::string mixMode = "balanced";  // 混合策略：balanced(默认,域均衡) | full(全量投喂)
    unsigned trainSeed = (unsigned)time(nullptr); // 采样/增强随机种子
    AugConfig g_aug;                   // 阶段E：数据增强配置（默认关闭）
    LrScheduleConfig lrSched;          // One-Cycle LR 调度（默认关闭，固定 LR）
    int c1Arg = 32;                    // 阶段0+1：Conv1 输出通道（--c1，默认 32）
    int c2Arg = 64;                    // 阶段0+1：Conv2 输出通道（--c2，默认 64）
    int depthArg = 2;                  // 阶段2：gen/bkc 新建模型的网络深度（--depth，默认 2）
    bool useProj = false;              // 阶段6：--res2 生成 1×1 投影残差网
    bool usePreact = false;            // 阶段6：--preact 生成 pre-activation 残差网

    // 集成融合 + TTA（test 用，默认与旧行为一致）
    std::string fuseModeStr = "soft";  // --fuse: soft|logit|geom|vote|gate|wbest
    float fuseMargin = 0.0f;           // --fuse_margin: gate 模式反选阈值
    float fuseTemp = 1.0f;             // --fuse_temp: 概率温度缩放
    std::vector<float> modelWeights;   // --weights: 每模型权重（空=全 1）
    bool ttaEnabled = false;           // --tta: 测试时增强
    int  ttaK = 5;                     // --tta_k: TTA 变换数量

    // 覆盖默认值
    try {
        if (args.count("epochs"))   epochs = std::stoi(args["epochs"]);
        if (args.count("lr_fc"))    lr_fc = std::stof(args["lr_fc"]);
        if (args.count("lr_conv1")) lr_conv1 = std::stof(args["lr_conv1"]);
        if (args.count("lr_conv2")) lr_conv2 = std::stof(args["lr_conv2"]);
        if (args.count("momentum")) momentum = std::stof(args["momentum"]);
        if (args.count("nesterov")) nesterov = true;
        if (args.count("model"))    modelPath = args["model"];
        // T5：保存格式开关。环境变量 CNN2_TXTMODEL（非空且非 "0"）→ 文本；命令行 --txtmodel 优先。
        {
            char* envTxt = nullptr; size_t envLen = 0;
            if (_dupenv_s(&envTxt, &envLen, "CNN2_TXTMODEL") == 0 && envTxt) {
                if (envTxt[0] != '\0' && std::string(envTxt) != "0") g_saveAcm = false;
                free(envTxt);
            }
        }
        if (args.count("txtmodel")) g_saveAcm = false;   // --txtmodel：强制文本格式
        if (args.count("batch"))    batch = std::stoi(args["batch"]);
        if (args.count("gpu"))      forceGpu = true;
        if (args.count("cpu"))      forceCpu = true;
        if (args.count("time"))     timeIt = true;
        if (args.count("log"))      logPath = args["log"];
        if (args.count("val"))      valDir = args["val"];
        if (args.count("wd"))       wdArg = std::stof(args["wd"]);
        if (args.count("epsave"))   { g_epsaveEnabled = true; epSaveArg = args["epsave"]; }
        if (args.count("epsn")) {
            int epsnArg = std::stoi(args["epsn"]);
            if (epsnArg < 1) std::cerr << "警告: --epsn " << epsnArg << " 非法（应 >= 1），已按 1 处理\n";
            g_epsn = (std::max)(1, epsnArg);
        }
        if (args.count("strict-data")) strictData = true;
        if (args.count("threads")) {
            try { nThreads = std::stoi(args["threads"]); }
            catch (...) { std::cerr << "无效 --threads\n"; return 1; }
        }
        if (args.count("outerror")) outError = true;
        if (args.count("P"))        g_modelSign = true;   // --P：保存模型时写完整签名
        if (args.count("BN")) g_useBN = true; // --BN/--bn：新建模型启用 BN（bn 已归一化为 BN）
        if (args.count("res"))      g_useRes = true;              // --res：新建残差网络（阶段3）
        if (args.count("res2"))     useProj = true;               // 阶段6：1×1 投影残差网
        if (args.count("preact"))   usePreact = true;             // 阶段6：pre-activation 残差网
        if (args.count("c1"))       c1Arg = std::stoi(args["c1"]); // 阶段0+1：Conv1 通道
        if (args.count("c2"))       c2Arg = std::stoi(args["c2"]); // 阶段0+1：Conv2 通道
        if (args.count("depth"))    depthArg = std::stoi(args["depth"]); // 阶段2：网络深度

        if (args.count("no_conv1")) train_conv1 = false;
        if (args.count("no_conv2")) train_conv2 = false;
        if (args.count("no_fc"))    train_fc = false;

        // One-Cycle 学习率调度（默认关闭）
        if (args.count("onecycle"))  lrSched.oneCycle = true;
        if (args.count("lr_warmup")) lrSched.warmupEpochs = std::stoi(args["lr_warmup"]);
        if (args.count("lr_min"))    lrSched.minFactor = std::stof(args["lr_min"]);
        if (args.count("lr_peak"))   lrSched.peakFactor = std::stof(args["lr_peak"]);

        // 混合训练 + 数据增强
        if (args.count("data2"))    data2 = args["data2"];

        // --mix <balanced|full>；--full 为 --mix full 的简写（大小写不敏感）
        if (args.count("mix"))      mixMode = args["mix"];
        if (args.count("full"))     mixMode = "full";
        for (char& c : mixMode) c = (char)std::tolower((unsigned char)c);
        if (mixMode != "balanced" && mixMode != "full") {
            std::cerr << "未知 --mix 模式: " << mixMode
                << "（可选 balanced | full）\n";
            return 1;
        }

        if (args.count("seed"))     trainSeed = (unsigned)std::strtoul(args["seed"].c_str(), nullptr, 10);
        g_aug = augConfigFromArgs(args);

        // 集成融合 + TTA
        if (args.count("fuse"))        fuseModeStr = args["fuse"];
        if (args.count("fuse_margin")) fuseMargin = std::stof(args["fuse_margin"]);
        if (args.count("fuse_temp"))   fuseTemp = std::stof(args["fuse_temp"]);
        if (args.count("weights"))     parseFloatList(args["weights"], modelWeights);
        if (args.count("tta"))         ttaEnabled = true;
        if (args.count("tta_k"))       ttaK = std::stoi(args["tta_k"]);

        // 预处理配置（--prep/--prep_target/--prep_stretch/--prep_ink/--prep_deskew_max/--diag）
        g_prep = prepConfigFromArgs(args);
    }
    catch (std::exception& e) {
        std::cerr << "参数解析失败: " << e.what() << "\n";
        return 1;
    }

    // T7：未显式 --model 时，默认模型名随格式（默认 acm），使无参 gen 默认产出 model.acm。
    if (modelPath.empty())
        modelPath = g_saveAcm ? std::string("model.acm") : std::string("model.txt");

    // ---------- 阶段B：互斥/冲突校验（B5，仅警告不退出，保持既有行为） ----------
    if (args.count("gpu") && args.count("cpu"))
        std::cerr << "警告: --gpu 与 --cpu 同时给出，将以 --cpu 为准（保持原有 CPU 优先行为）。\n";
    {
        int resCnt = (int)(args.count("res") + args.count("res2") + args.count("preact"));
        if (resCnt >= 2)
            std::cerr << "警告: --res/--res2/--preact 同时给出 " << resCnt
                      << " 个，将按优先级 res2 > preact > res 择一。\n";
    }
    if (args.count("prep") && args["prep"] == "1")
        std::cerr << "警告: --prep 未带值，将关闭全部预处理（如需如此请显式写 --prep none）。\n";

    // ---------- 阶段B：gen 模式 --out 作为 --model 别名（B6） ----------
    // 仅 gen 生效，且未显式给出 --model 时才应用，避免覆盖 --model。
    if (mode == "gen" && args.count("out") && !args.count("model"))
        modelPath = args["out"];

    // 阶段4.1：把已解析的 --seed 同步到全局，供 buildKernels（初始化）与训练洗牌使用。
    // 未指定 --seed 时 g_seedSet=false，保留时间/系统随机（默认随机行为不变）。
    g_seedSet = (args.count("seed") != 0);
    g_seed = trainSeed;

    // 阶段4.4：同步独立验证集状态。未传 --val 时 g_valEnabled=false，默认行为不变。
    g_valDir = valDir;
    g_valEnabled = !g_valDir.empty();
    g_saveBest = g_valEnabled;                 // --val 时默认保存 <model>.best
    // T7：<model>.best 的后缀随模型格式（保留原目录，形如 <stem>.best.acm / .txt）。
    {
        const ModelFormat bf = modelSaveFormat(modelPath);
        std::string bdir;
        size_t bsep = modelPath.find_last_of("/\\");
        if (bsep != std::string::npos) bdir = modelPath.substr(0, bsep + 1);
        g_bestPath = bdir + pathStem(modelPath) + ".best"
            + (bf == ModelFormat::Acm ? ".acm" : ".txt");
    }
    g_bestValAcc = -1.0f;

    // ---- One-Cycle 参数合法性：非法值 clamp 并告警（保证公式有定义） ----
    if (lrSched.warmupEpochs < 0) {
        std::cerr << "警告: --lr_warmup 不能为负，已改为自动(epochs/5)\n";
        lrSched.warmupEpochs = 0;
    }
    if (lrSched.minFactor < 0.0f) {
        std::cerr << "警告: --lr_min 不能为负，已置 0\n";
        lrSched.minFactor = 0.0f;
    }
    if (lrSched.peakFactor <= 0.0f) {
        std::cerr << "警告: --lr_peak 必须为正，已置 1.0\n";
        lrSched.peakFactor = 1.0f;
    }
    if (lrSched.minFactor > lrSched.peakFactor)
        lrSched.minFactor = lrSched.peakFactor;

    // ---- 融合模式字符串 -> 枚举 ----
    FuseMode fuseMode = FUSE_SOFT;
    if      (fuseModeStr == "soft")  fuseMode = FUSE_SOFT;
    else if (fuseModeStr == "logit") fuseMode = FUSE_LOGIT;
    else if (fuseModeStr == "geom")  fuseMode = FUSE_GEOM;
    else if (fuseModeStr == "vote")  fuseMode = FUSE_VOTE;
    else if (fuseModeStr == "gate")  fuseMode = FUSE_GATE;
    else if (fuseModeStr == "wbest") fuseMode = FUSE_WBEST;
    else {
        std::cerr << "未知 --fuse 模式: " << fuseModeStr
            << "（可选 soft|logit|geom|vote|gate|wbest）\n";
        return 1;
    }
    if (fuseTemp <= 0.0f) fuseTemp = 1.0f;
    if (ttaK < 1) ttaK = 1;

    std::cout << "路径: " << path << "\n";
    std::cout << "模式: " << mode << "\n";
    std::cout << "epochs = " << epochs << "\n";
    std::cout << "lr_fc = " << lr_fc << "\n";
    std::cout << "lr_conv1 = " << lr_conv1 << "\n";
    std::cout << "lr_conv2 = " << lr_conv2 << "\n";
    std::cout << "onecycle = " << (lrSched.oneCycle ? "on" : "off")
        << " warmup = " << (lrSched.warmupEpochs > 0 ? std::to_string(lrSched.warmupEpochs) : std::string("auto"))
        << " lr_min = " << lrSched.minFactor
        << " lr_peak = " << lrSched.peakFactor << "\n";
    std::cout << "momentum = " << momentum
        << " nesterov = " << (nesterov ? 1 : 0) << "\n";
    std::cout << "train_conv1 = " << train_conv1
        << " train_conv2 = " << train_conv2
        << " train_fc = " << train_fc << "\n";
    std::cout << "batch = " << batch
        << " force_gpu = " << forceGpu
        << " force_cpu = " << forceCpu << "\n";
    std::cout << "计时(--time) = " << (timeIt ? "on" : "off") << "\n";
    std::cout << "模型签名(--P) = " << (g_modelSign ? "on" : "off") << "\n";
    if (!logPath.empty()) std::cout << "训练日志(--log) = " << logPath << "\n";
    if (!valDir.empty())  std::cout << "验证集(--val) = " << valDir << "\n";
    if (wdArg != 0.0f)    std::cout << "weight_decay(--wd) = " << wdArg << "\n";
    std::cout << "模型文件: " << modelPath << "\n";
    std::cout << "预处理: crop=" << g_prep.doCrop
        << " deskew=" << g_prep.doDeskew
        << " resize=" << g_prep.doResize
        << " center=" << g_prep.doCenter
        << " thick=" << g_prep.doThick
        << " target=" << g_prep.target
        << " stretch=" << g_prep.stretch
        << " ink=" << g_prep.targetInk
        << " deskew_max=" << g_prep.deskewMaxDeg
        << " diag=" << g_prep.diag << "\n";
    std::cout << "data2 = " << (data2.empty() ? std::string("(none)") : data2)
        << " seed=" << trainSeed << "\n";
    std::cout << "混合策略: " << mixMode
        << (mixMode == "full" ? "（全量投喂，不下采样/均衡）"
                              : "（域均衡分层采样）") << "\n";
    std::cout << "数据增强: " << (g_aug.enable ? "on" : "off")
        << " k=" << g_aug.copies
        << " shift=" << g_aug.shift
        << " rotate=" << g_aug.rotateDeg
        << " scale=" << g_aug.scaleJitter
        << " elastic=" << g_aug.elasticAlpha
        << " sigma=" << g_aug.elasticSigma
        << " thick=" << g_aug.thickPct << "\n";
    std::cout << "集成融合: " << fuseModeName(fuseMode)
        << " margin=" << fuseMargin
        << " temp=" << fuseTemp
        << " weights=";
    if (modelWeights.empty()) {
        std::cout << "(all 1)";
    }
    else {
        for (size_t i = 0; i < modelWeights.size(); ++i)
            std::cout << (i ? "," : "") << modelWeights[i];
    }
    std::cout << " TTA=" << (ttaEnabled ? "on" : "off")
        << " tta_k=" << (ttaEnabled ? ttaClampK(ttaK) : 0) << "\n";

    // ---------- 阶段0+1：依据模式/权重文件确定运行时规格 g_net ----------
    {
        std::string perr;
        // 阶段6：统一按 CLI 选择残差类型构造新网络规格（gen / bkc 新建共用）。
        auto makeCliSpec = [&]() -> NetSpec {
            if (useProj)   return makeProjResNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg);
            if (usePreact) return makePreactResNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg);
            if (g_useRes)  return makeResNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg);
            return makeDefaultNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg, c2Arg);
        };
        if (mode == "gen") {
            // 生成：用 CLI --c1/--c2/--depth 构造规格；--BN 决定 bn；--res/--res2/--preact 选残差类型。
            // depth==2 且无残差时与阶段1 默认完全一致；depth>2 用默认「同构 conv→pool」schedule。
            if (useProj || usePreact || g_useRes) {
                if (depthArg < 2) { std::cerr << "残差网络要求 --depth >= 2\n"; return 1; }
                g_net = makeCliSpec();
                std::cout << "残差网络: 全部层通道 c=" << c1Arg
                    << "，类型=" << (useProj ? "1x1 投影" : usePreact ? "pre-activation" : "identity")
                    << "，末层残差指向第 1 层\n";
            } else {
                g_net = makeDefaultNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg, c2Arg);
            }
        }
        else if (mode == "bkc") {
            // 训练：若 --model 已存在则先预览文件维度（以文件为准），否则用 CLI 维度
            std::ifstream probe(modelPath);
            bool hasModel = probe.good();
            probe.close();
            NetSpec spec;
            if (hasModel && peekModelDims(modelPath, spec, &perr)) {
                g_net = spec;
                if (spec.depth >= 2 &&
                    (spec.layer[0].outC != c1Arg || spec.layer[1].outC != c2Arg)) {
                    std::cerr << "警告: --model 文件维度 c1=" << spec.layer[0].outC
                        << " c2=" << spec.layer[1].outC << " 与 CLI --c1/--c2 ("
                        << c1Arg << "/" << c2Arg << ") 冲突，以文件为准\n";
                }
            }
            else {
                if (useProj || usePreact || g_useRes) {
                    if (depthArg < 2) { std::cerr << "残差网络要求 --depth >= 2\n"; return 1; }
                    g_net = makeCliSpec();
                    std::cout << "残差网络: 全部层通道 c=" << c1Arg
                        << "，类型=" << (useProj ? "1x1 投影" : usePreact ? "pre-activation" : "identity")
                        << "，末层残差指向第 1 层\n";
                } else {
                    g_net = makeDefaultNetSpec(32, 32, 10, g_useBN, depthArg, c1Arg, c2Arg);
                }
            }
        }
        else if (mode == "test") {
            // 测试：逐个预览模型，要求维度一致（异构集成留到阶段3）
            std::vector<std::string> mp = splitModelList(modelPath);
            if (mp.empty()) { std::cerr << "test: 未指定模型文件\n"; return 1; }
            NetSpec spec;
            if (!peekModelDims(mp[0], spec, &perr)) {
                std::cerr << "test: 无法读取模型维度 " << mp[0]
                    << "（" << perr << "）\n";
                return 1;
            }
            for (size_t m = 1; m < mp.size(); ++m) {
                NetSpec s2;
                if (!peekModelDims(mp[m], s2, &perr)) {
                    std::cerr << "test: 无法读取模型维度 " << mp[m]
                        << "（" << perr << "）\n";
                    return 1;
                }
                if (s2.bn != spec.bn) {
                    std::cerr << "集成: 模型 BN/非BN 混合不被支持（" << mp[m]
                        << " 与 " << mp[0] << " 不一致）\n";
                    return 1;
                }
                bool dimsSame = (s2.depth == spec.depth &&
                                 s2.flat == spec.flat && s2.classes == spec.classes);
                for (int li = 0; li < spec.depth && dimsSame; ++li) {
                    if (s2.layer[li].outC != spec.layer[li].outC ||
                        s2.layer[li].inC != spec.layer[li].inC ||
                        s2.layer[li].kH   != spec.layer[li].kH   ||
                        s2.layer[li].kW   != spec.layer[li].kW   ||
                        s2.layer[li].stride != spec.layer[li].stride ||
                        s2.layer[li].pad    != spec.layer[li].pad    ||
                        s2.layer[li].pool   != spec.layer[li].pool   ||
                        s2.layer[li].res    != spec.layer[li].res    ||
                        s2.layer[li].resFrom != spec.layer[li].resFrom ||
                        s2.layer[li].resPos  != spec.layer[li].resPos) dimsSame = false;
                }
                if (!dimsSame) {
                    std::cerr << "集成: 模型维度不一致，异构集成暂不支持（阶段3）: "
                        << mp[m] << "\n";
                    return 1;
                }
            }
            g_net = spec;
        }
        else {
            // 单图推理 / 默认：由 --model 预览维度
            NetSpec spec;
            if (!peekModelDims(modelPath, spec, &perr)) {
                std::cerr << "无法读取模型维度 " << modelPath
                    << "（" << perr << "）\n";
                return 1;
            }
            g_net = spec;
        }
        std::string verr;
        if (!validateNetSpec(g_net, &verr)) {
            std::cerr << "网络规格非法: " << verr << "\n";
            return 1;
        }

        // ---- BN 与 CLI 一致性检测（peek 后、buildKernels 前） ----
        // modelBn 取最终规格（gen=CLI、bkc/test/单图=模型文件），cliBn 取 --BN。
        if (!checkBnCliAgreement(g_net.bn, g_useBN, mode.c_str())) return 1;

        // ---- P2：真 BN 放行规则（非 BN 路径完全不进入本块，逐位不变） ----
        // 允许：gen --BN；bkc --gpu + BN（GPU 真 BN 训练，跨 batch 统计、γ/β 可训练）；
        //       test --gpu + BN（eval 模式：BN 用 running 统计量，不更新）；
        //       单图/GUI 用 CPU 推理 BN 模型（forwardConv 已含 bn_infer）。
        // 禁止：bkc + BN 但最终走 CPU（CPU 训练不支持 BN）。
        if (g_net.bn && mode == "bkc") {
            const bool willUseCpu = forceCpu || (!forceGpu && !gpuAvailable());
            if (willUseCpu) {
                std::cerr << "CPU 训练不支持 BN，请用 --gpu\n";
                return 1;
            }
        }
    }

    buildKernels();

    int flatSize = g_net.flat;
    std::cout << "展平大小: " << flatSize << "\n";
    std::cout << "参数总量: " << g_net.total << "\n";

    FullyConnected fc(flatSize, g_net.classes);

    // ========== gen ==========
    if (mode == "gen") {
        // 阶段4.1：FC 随机初始化与卷积核同源于 --seed（独立子种子），保证 gen 可复现；
        // 未指定 --seed 时沿用时间种子（默认随机）。
        srand(g_seedSet ? mixSeed(g_seed, 0x2u) : (unsigned)time(nullptr));
        fc.randomInit();
        if (!saveModel(modelPath, fc)) return 1;
        std::cout << "已生成 " << modelPath << "\n";
        return 0;
    }

    // ========== bkc ==========
    if (mode == "bkc") {
        // 功能二：--epsave 每轮存档目录（默认 EPTSET\<模型名>\，相对 CWD；末尾补 '\'）。
        if (g_epsaveEnabled) {
            g_epSaveDir = (epSaveArg.empty() || epSaveArg == "1")
                ? ("EPTSET\\" + modelBaseName(modelPath) + "\\")
                : epSaveArg;
            if (!g_epSaveDir.empty() &&
                g_epSaveDir.back() != '\\' && g_epSaveDir.back() != '/')
                g_epSaveDir += '\\';
            if (!ensureDirectory(g_epSaveDir)) {
                std::cerr << "无法创建 --epsave 目录: " << g_epSaveDir << "\n";
                return 1;
            }
            std::cout << "每轮存档(--epsave) = " << g_epSaveDir
                << " (epsn=" << g_epsn << ")\n";
        }
        // 阶段4.3：--log 打开训练日志（追加模式；新文件写表头）。未传 --log 时不建立/不写。
        if (!logPath.empty()) {
            bool newFile = true;
            {
                std::ifstream probe(logPath, std::ios::binary | std::ios::ate);
                if (probe) { newFile = (probe.tellg() == 0); probe.close(); }
            }
            g_log.open(logPath, std::ios::app);
            if (!g_log) { std::cerr << "无法打开 --log 文件: " << logPath << "\n"; return 1; }
            if (newFile) g_log << (g_valEnabled ? kLogHeaderVal : kLogHeader);
            g_log.flush();
            g_logEnabled = true;
        }
        {
            std::ifstream test(modelPath);
            if (test.good()) {
                test.close();
                if (!loadModel(modelPath, fc)) return 1;
                // H4：加载后校验权重形状与运行时规格 g_net 自洽，与 test 路径一致
                if (!checkModelDims(modelPath, fc)) return 1;
                std::cout << "已从 " << modelPath << " 加载模型，继续训练\n";
            }
            else {
                // 阶段4.1：新建模型的 FC 初始化同样由 --seed 派生，保证 bkc 可复现。
                srand(g_seedSet ? mixSeed(g_seed, 0x2u) : (unsigned)time(nullptr));
                fc.randomInit();
                std::cout << "未找到 " << modelPath << "，使用随机初始化\n";
            }
        }

        // 构建训练样本（支持 --data2 混合；--mix balanced 分层均衡 / full 全量投喂）
        bool fullMix = (mixMode == "full");
        std::vector<Sample> samples = buildTrainSet(path, data2, trainSeed, fullMix);

        // 增强用随机引擎（与采样种子分离，保证可复现）
        std::mt19937 augRng(trainSeed ^ CNN_GOLDEN_RATIO);

        // 阶段4.4：--strict-data 时先对训练目录（及 --val 目录）做数据校验，发现非法即退出。
        if (strictData) {
            int bad = checkDataDir(path, true);
            if (g_valEnabled) bad += checkDataDir(g_valDir, true);
            if (bad > 0) return 1;
        }

        // 阶段4.4：加载独立验证集（--val）。与 test 同源预处理（g_prep），不做增强。
        // val==train 时明确告警：指标只能当占位，不能用于评估泛化。
        if (g_valEnabled) {
            if (normalizeDir(g_valDir) == normalizeDir(path))
                std::cout << "警告: --val 与训练目录相同（val=train），验证指标仅占位、不代表泛化\n";
            if (!loadValSet(g_valDir))
                std::cout << "警告: --val 目录无可用验证样本: " << g_valDir << "\n";
            else
                std::cout << "验证样本数: " << g_valLabels.size() << "\n";
        }

        // ========== GPU dispatch (new) ==========
        {
            bool useGpu;
            if (forceCpu) useGpu = false;
            else if (forceGpu) useGpu = true;
            else useGpu = gpuAvailable();

            if (useGpu && !gpuAvailable()) {
                std::cerr << "--gpu requested but no CUDA device available\n";
                return 1;
            }

            if (useGpu) {
                // 阶段2：GPU kernel 已支持任意规模（运行时 NetSpec），无需再拦截。
                std::cout << "GPU training enabled, batch = " << batch << "\n";

                std::vector<float> data;
                std::vector<int> labels;
                // 数据加载线程数归一化：0=auto(hardware_concurrency)；下限 1；上限样本数
                int th = (nThreads == 0) ? (int)std::thread::hardware_concurrency() : nThreads;
                if (th <= 0) th = 1;
                th = (int)std::min<size_t>((size_t)th,
                    samples.empty() ? (size_t)1 : samples.size());
                std::cout << "并行线程数: " << th << "\n";
                // 加载用种子基（与 augRng 同源，保证固定 seed 可复现）
                unsigned loadSeedBase = trainSeed ^ CNN_GOLDEN_RATIO;
                // --time：数据加载（读图 + 预处理 + 增强准备）计时起点
                auto tData0 = std::chrono::steady_clock::now();
                std::cout << "原始训练样本数: " << samples.size()
                    << "，增强副本/图: "
                    << (g_aug.enable ? g_aug.copies : 0) << "\n";
                if (!loadAllDataSamples(samples, g_aug, augRng, data, labels,
                    th, timeIt, loadSeedBase)) {
                    std::cerr << "no valid training data\n";
                    return 1;
                }
                std::cout << "训练规模: 原始 " << samples.size()
                    << " -> 扩增后 " << labels.size() << " 样本\n";
                if (timeIt) {
                    double loadSec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tData0).count();
                    std::cout << "数据加载耗时: " << formatDur(loadSec) << "\n";
                }
                // 注：windows.h 定义了 min 宏，这里用三元表达式避免 (std::min) 陷阱
                int effBatch = labels.empty()
                    ? batch
                    : ((batch > 0 && (size_t)batch < labels.size())
                        ? batch : (int)labels.size());
                printGpuMemoryEstimate(labels.size(), effBatch);

                std::vector<float> params;
                flattenParams(params, fc);

                GpuTrainConfig cfg;
                cfg.epochs = epochs;
                cfg.batch = batch;
                cfg.lr_fc = lr_fc;
                cfg.lr_conv1 = lr_conv1;
                cfg.lr_conv2 = lr_conv2;
                cfg.train_conv1 = train_conv1;
                cfg.train_conv2 = train_conv2;
                cfg.train_fc = train_fc;
                cfg.momentum = momentum;
                cfg.nesterov = nesterov;
                cfg.weightDecay = wdArg;   // L2 weight decay（默认 0.0）
                cfg.timeIt = timeIt;   // GPU 内部按 epoch 打印耗时
                cfg.seed = trainSeed;  // 阶段4.1：GPU 每 epoch 确定性洗牌种子
                cfg.lrSched = lrSched; // One-Cycle LR 调度（默认关闭）

                // 跨平台计时：gpuTrain 末尾已 cudaDeviceSynchronize，返回时 GPU 已完成
                auto tGpu0 = std::chrono::steady_clock::now();
                if (!gpuTrain(data, labels, cfg, params, gpuEpochEndCallback, &fc, g_net)) {
                    std::cerr << "GPU training failed\n";
                    return 1;
                }

                scatterParams(params.data(), fc);
                saveModel(modelPath, fc);
                std::cout << "saved " << modelPath << "\n";
                logCheckpoint(epochs - 1, modelPath);   // 阶段4.3：结束存档元数据
                if (timeIt) {
                    auto tEnd = std::chrono::steady_clock::now();
                    double trainSec = std::chrono::duration<double>(tEnd - tGpu0).count();
                    double totalSec = std::chrono::duration<double>(tEnd - tData0).count();
                    std::cout << "总训练时间: " << formatDur(totalSec)
                        << "（数据加载 " << formatDur(totalSec - trainSec)
                        << "，训练 " << formatDur(trainSec) << "）\n";
                }
                return 0;
            }
        }
        // ========== end GPU dispatch ==========

        // CPU 路径：初始化 momentum velocity 缓冲（初值 0，训练全程持续累积）
        resetVelocity(fc);

        // 阶段4.1：每 epoch 用 (seed, epoch) 派生的确定性引擎洗牌（替换 random_device）。
        std::mt19937 g;

        // 跨平台计时：CPU 训练起点（仅 --time 开启时用于输出）
        auto tCpu0 = std::chrono::steady_clock::now();
        double totalLoadSec = 0.0;   // --time：全程数据加载（读图+预处理+增强）累计

        // one-cycle（CPU）：按样本粒度计算 lr（step = epoch*N + 样本序号）
        const long long cpuN = (long long)samples.size();
        const long long cpuTotalSteps = (long long)epochs * cpuN;
        const int cpuWarmEpochs = (lrSched.warmupEpochs > 0)
            ? lrSched.warmupEpochs : (std::max)(1, epochs / 5);
        const long long cpuWarmupSteps = (long long)cpuWarmEpochs * cpuN;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto tEp0 = std::chrono::steady_clock::now();
            // 阶段4.1：同 seed 同 epoch → 同一洗牌顺序（与 GPU 用同一 mixSeed 方案）。
            g.seed(mixSeed(trainSeed, 0x5EEDu ^ (unsigned)epoch));
            std::shuffle(samples.begin(), samples.end(), g);

            float totalLoss = 0.0f;
            int correct = 0, total = 0;
            std::vector<int> predCount(g_net.classes, 0);   // H2：运行时类别数
            double epLoadSec = 0.0;   // --time：本 epoch 数据加载（读图+预处理+增强）累计耗时
            long long cpuSampleIdx = 0;   // one-cycle：本 epoch 内样本序号

            for (auto& s : samples) {
                // one-cycle：本样本的全局 step 与 lr 倍率（默认恒 1.0f）。
                const long long cpuStep = (long long)epoch * cpuN + cpuSampleIdx;
                ++cpuSampleIdx;
                const float lrF = lrScaleFactor(lrSched, cpuStep, cpuTotalSteps, cpuWarmupSteps);
                std::chrono::steady_clock::time_point tLoad0;
                if (timeIt) tLoad0 = std::chrono::steady_clock::now();
                ConvMap input(1, 32, 32);
                if (!loadImagePreprocessed(s.path, input)) continue;

                // CPU 路径：每 epoch 每样本训练前在线随机增强
                if (g_aug.enable) augmentRandom(input, g_aug, augRng);
                if (timeIt) {
                    epLoadSec += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tLoad0).count();
                }

                float loss = 0.0f;
                int pred = trainOne(input, s.label, fc,
                    lr_fc    * lrF,
                    lr_conv1 * lrF,
                    lr_conv2 * lrF,
                    momentum, nesterov,
                    train_conv1, train_conv2, train_fc,
                    wdArg,
                    loss);

                totalLoss += loss;
                if (pred == s.label) ++correct;
                ++total;
                if (pred >= 0 && pred < (int)predCount.size()) predCount[pred]++;
            }

            if (total == 0) { std::cerr << "没有可用训练文件\n"; break; }

            // 阶段4.3：本 epoch 统计量先落局部变量，stdout 与 --log 共用，保证同源。
            const float cpuLoss = totalLoss / total;
            const float cpuAcc  = (float)correct / total;
            // 本 epoch 末（最后一个样本）的实际 lr（非 one-cycle 时 cpuFEp=1.0f）。
            const long long cpuEpReportStep = (long long)epoch * cpuN
                + (cpuN > 0 ? cpuN - 1 : 0);
            const float cpuFEp = lrScaleFactor(lrSched, cpuEpReportStep,
                                               cpuTotalSteps, cpuWarmupSteps);
            const float cpuLr1  = lr_conv1 * cpuFEp;
            const float cpuLr2  = lr_conv2 * cpuFEp;
            const float cpuLrFc = lr_fc    * cpuFEp;

            std::cout << "epoch " << epoch
                << " loss " << cpuLoss
                << " acc " << cpuAcc;
            // 仅 one-cycle 开启时追加 lr，保证默认 stdout 与改动前逐字节一致
            if (lrSched.oneCycle) {
                std::cout << " lr(c1/c2/fc)=" << cpuLr1
                    << "/" << cpuLr2
                    << "/" << cpuLrFc;
            }
            std::cout << "\n";

            // 本 epoch 耗时：--time 打印或 --log 落盘都需要，故一并计算（读时钟无副作用）。
            double epSec = 0.0;
            if (timeIt || g_logEnabled) {
                epSec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tEp0).count();
            }
            if (timeIt) {
                totalLoadSec += epLoadSec;
                std::cout << "epoch " << epoch << " 耗时: " << formatDur(epSec)
                    << "（数据加载 " << formatDur(epLoadSec)
                    << "，计算 " << formatDur(epSec - epLoadSec) << "）\n";
            }
            // 阶段4.4：--val 时本 epoch 末在验证集上评估（只读前向，不改变训练状态）。
            float valAcc = -1.0f;
            bool  hasVal = false;
            if (g_valEnabled) {
                valAcc = runValEvalAndBest(fc, epoch, g_bestPath);
                hasVal = (valAcc >= 0.0f);
            }
            logEpochMetrics(epoch, cpuLoss, cpuAcc, cpuLr1, cpuLr2, cpuLrFc, epSec,
                            valAcc, hasVal);

            std::cout << "预测分布: ";
            for (int i = 0; i < (int)predCount.size(); ++i)
                std::cout << i << ":" << predCount[i] << " ";
            std::cout << "\n";

            // 每 10 轮保存一个中间模型
            if (epoch % 10 == 9) {
                std::string epPath = withModelExt("model_epoch" + std::to_string(epoch), currentSaveFormat());
                saveModel(epPath, fc);
                std::cout << "已保存 " << epPath << "\n";
                logCheckpoint(epoch, epPath);   // 阶段4.3：存档元数据
            }
            // 功能二：--epsave 每轮存档（与上面每 10 轮存档并存、互不影响）。
            if (g_epsaveEnabled && g_epsn > 0 && (epoch % g_epsn == 0)) {
                std::string epPath = g_epSaveDir + withModelExt("ep_" + std::to_string(epoch), currentSaveFormat());
                saveModel(epPath, fc);
                std::cout << "已保存 " << epPath << "\n";
                logCheckpoint(epoch, epPath);
            }
        }

        if (timeIt) {
            double sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tCpu0).count();
            std::cout << "总训练时间: " << formatDur(sec)
                << "（数据加载累计 " << formatDur(totalLoadSec)
                << "，训练 " << formatDur(sec - totalLoadSec) << "）\n";
        }

        saveModel(modelPath, fc);
        std::cout << "已保存 " << modelPath << "\n";
        logCheckpoint(epochs - 1, modelPath);   // 阶段4.3：结束存档元数据
        return 0;
    }

    // ========== test ==========
    if (mode == "test") {
        // --model 支持逗号分隔多模型：逐个加载并扁平化为独立参数副本，
        // 推理时对各模型 softmax 概率逐类平均做集成（M=1 时行为与旧版一致）。
        std::vector<std::string> modelPaths = splitModelList(modelPath);
        if (modelPaths.empty()) {
            std::cerr << "test: 未指定模型文件\n";
            return 1;
        }

        std::vector<std::vector<float>> paramsList;
        paramsList.reserve(modelPaths.size());
        for (size_t m = 0; m < modelPaths.size(); ++m) {
            if (!loadModel(modelPaths[m], fc)) {
                std::cerr << "集成: 无法加载模型 " << modelPaths[m] << "\n";
                return 1;
            }
            if (!checkModelDims(modelPaths[m], fc)) return 1; // 维度不符直接退出
            std::vector<float> p;
            flattenParams(p, fc);          // 保存该模型在当前布局下的完整参数
            paramsList.push_back(std::move(p));
            std::cout << "已加载模型[" << m << "]: " << modelPaths[m] << "\n";
        }
        const int M = (int)modelPaths.size();
        std::cout << "模型数: " << M;
        if (M > 1) std::cout << "（" << fuseModeName(fuseMode) << " 融合集成）";
        std::cout << "\n";

        // 归一化每模型权重到长度 M（未提供则全 1；提供的多则截断）
        std::vector<float> w = modelWeights;
        if ((int)w.size() != M) w.resize(M, 1.0f);

        // 测试后端：默认 CPU（向后兼容）；--gpu 显式启用批量 GPU 推理；
        // --cpu 与 --gpu 同时给出时以 --cpu 为准，强制 CPU。
        bool useGpu = forceGpu && !forceCpu;
        if (useGpu && !gpuAvailable()) {
            std::cerr << "--gpu requested but no CUDA device available\n";
            return 1;
        }
        // 阶段2：GPU 推理 kernel 已支持任意规模（运行时 NetSpec），无需再拦截。

        // ---------- GPU 批量推理 ----------
        if (useGpu) {
            std::cout << "测试后端: GPU（batch=" << batch << "）\n";

            std::vector<std::string> files = listTxtFiles(path);
            std::cout << "测试文件数: " << files.size() << "\n";

            // 预处理与训练/CPU test 完全共用 loadImagePreprocessed(g_prep)
            std::vector<float> data;
            std::vector<int> labels;
            std::vector<std::string> names;   // 与 labels 下标对齐（跳过无效文件后仍一一对应）
            for (auto& f : files) {
                int label = getLabelFromPath(f);
                if (label < 0) continue;

                ConvMap input(1, 32, 32);
                if (!loadImagePreprocessed(f, input)) continue;

                pushFlat(input, label, data, labels);
                names.push_back(f);           // 必须与 pushFlat 成对，保证 names[i] ↔ labels[i]
            }

            int N = (int)labels.size();
            std::cout << "GPU 有效测试样本: " << N << "\n";
            if (N == 0) { std::cerr << "没有可用测试文件\n"; return 1; }

            // 逐模型逐样本概率张量 allProb[(m*N+i)*C+c]，C = g_net.classes
            std::vector<float> allProb((size_t)N * M * g_net.classes, 0.0f);
            std::vector<int> pred;            // gpuPredict 的 soft-argmax（此处仅占位）

            if (!ttaEnabled) {
                if (!gpuPredict(data, N, paramsList, batch, pred,
                                &allProb, g_net)) {
                    std::cerr << "GPU 推理失败\n";
                    return 1;
                }
            }
            else {
                int T = ttaClampK(ttaK);
                std::cout << "TTA: on  k=" << T << "（log 概率空间平均）\n";
                std::vector<double> logitAcc((size_t)N * M * g_net.classes, 0.0);
                std::vector<float> dataT;
                std::vector<int> predT;
                std::vector<float> allProbT;
                int nc = 0; const TTATransform* cand = ttaCandidates(nc);
                for (int t = 0; t < T; ++t) {
                    const std::vector<float>* pData = &data;   // t0 原图零拷贝
                    if (cand[t].dx != 0 || cand[t].dy != 0 || cand[t].deg != 0.0f) {
                        makeTTADataForTransform(data, N, t, dataT);
                        pData = &dataT;
                    }
                    allProbT.assign((size_t)N * M * g_net.classes, 0.0f);
                    predT.clear();
                    if (!gpuPredict(*pData, N, paramsList, batch, predT,
                                    &allProbT, g_net)) {
                        std::cerr << "GPU 推理失败 (TTA t=" << t << ")\n";
                        return 1;
                    }
                    // 越界防御：取两缓冲区较小长度（C 一致时相等）
                    const size_t kMax = (std::min)(allProbT.size(), logitAcc.size());
                    for (size_t k = 0; k < kMax; ++k)
                        logitAcc[k] += std::log((double)allProbT[k] + 1e-9);
                }
                logitAccToProb(logitAcc, N, M, T, allProb);
            }

            int correct = 0, total = 0, wrong = 0;
            std::vector<int> pc(g_net.classes, 0), pt(g_net.classes, 0);
            const int M_gpu = (int)paramsList.size();   // 模型数（逐模型概率平均的分母）
            std::ofstream errFile;
            if (outError) {
                errFile.open("outerror.txt");
                std::cout << "\n===== 判错样本清单 (--outerror) =====\n";
                if (errFile) errFile << "===== 判错样本清单 =====\n";
                if (errFile) errFile << "模式: GPU  模型数: " << M_gpu
                                     << "  样本数: " << N << "\n";
            }
            for (int i = 0; i < N; ++i) {
                int label = labels[i];
                int p = fusePredictSample(allProb.data(), N, M, i, w,
                                          fuseMode, fuseMargin, fuseTemp);
                if (p == label) ++correct;
                ++total;
                if (label >= 0 && label < (int)pt.size()) {
                    pt[label]++;
                    if (p == label) pc[label]++;
                }

                if (outError && p != label) {
                    ++wrong;
                    float conf  = modelWeightedProb(allProb.data(), N, M, i, w, p);
                    float tprob = modelWeightedProb(allProb.data(), N, M, i, w, label);
                    std::ostringstream oss;
                    oss << "[错] " << names[i]
                        << "  真值=" << label
                        << "  预测=" << p
                        << "  置信度=" << conf
                        << "  真值概率=" << tprob;
                    std::string line = oss.str();
                    std::cout << line << "\n";
                    if (errFile) errFile << line << "\n";
                }
            }
            if (outError) {
                std::cout << "判错总数: " << wrong << " / " << total << "\n";
                if (errFile) errFile << "判错总数: " << wrong << " / " << total << "\n";
                if (wrong == 0) {
                    std::cout << "（未发现判错样本）\n";
                    if (errFile) errFile << "（未发现判错样本）\n";
                }
            }

            std::cout << "标准准确率: " << (float)correct / total
                << " (" << correct << "/" << total << ")\n";
            for (int i = 0; i < (int)pt.size(); ++i) {
                if (pt[i] == 0) continue;
                std::cout << "数字 " << i << ": "
                    << (float)pc[i] / pt[i]
                    << " (" << pc[i] << "/" << pt[i] << ")\n";
            }
            return 0;
        }

        // ---------- CPU 推理（单/多模型统一；默认路径，单模型行为不变） ----------
        // 预处理只做一次并缓存输入，随后对每个模型跑一遍全测试集，把每个模型
        // 对每个样本的 softmax 概率写入 allProb[(m*N+i)*C+c]（C=g_net.classes）；最后按 --fuse
        // 指定的融合方式逐样本定预测。（--tta 时先在 log 概率空间对变换求平均）
        std::vector<std::string> files = listTxtFiles(path);
        std::cout << "测试文件数: " << files.size() << "\n";

        // 线程数归一化（--threads，0=auto）：下限 1；加载段上限文件数、前向段上限样本数。
        // test 不做增强（无 RNG 交互）；模型参数在并行期间只读，聚合按样本独立。
        int th = (nThreads == 0) ? (int)std::thread::hardware_concurrency() : nThreads;
        if (th <= 0) th = 1;

        std::vector<float> data;      // N*1024 连续 0/1 输入（与 GPU 路径一致）
        std::vector<int> labels;      // N
        std::vector<std::string> names;   // N：与 labels 下标对齐（跳过无效文件后仍一一对应）

        if (th <= 1 || files.size() <= 1) {
            // legacy 单线程分支（--threads 1）：与改动前逐字节/逐位一致
            for (auto& f : files) {
                int label = getLabelFromPath(f);
                if (label < 0) continue;

                ConvMap input(1, 32, 32);
                if (!loadImagePreprocessed(f, input)) continue;

                pushFlat(input, label, data, labels);
                names.push_back(f);       // 必须与 pushFlat 成对，保证 names[i] ↔ labels[i]
            }
        }
        else {
            // 并行读图 + 预处理：每线程处理 files 的一个下标区间，写各自本地缓冲；
            // 主线程 join 后按线程顺序（= 文件顺序）合并，样本顺序与串行完全一致。
            struct TestLoadResult {
                std::vector<float>       data;
                std::vector<int>         labels;
                std::vector<std::string> names;
            };
            const size_t FN = files.size();
            const int T = (int)std::min<size_t>((size_t)th, FN);
            std::vector<TestLoadResult> results((size_t)T);
            std::vector<std::thread> loadWorkers;
            loadWorkers.reserve((size_t)T);
            for (int t = 0; t < T; ++t) {
                const size_t begin = FN * (size_t)t / (size_t)T;
                const size_t end   = FN * (size_t)(t + 1) / (size_t)T;
                loadWorkers.emplace_back([&, t, begin, end]() {
                    TestLoadResult& out = results[(size_t)t];
                    out.data.reserve((end - begin) * 1024);
                    out.labels.reserve(end - begin);
                    out.names.reserve(end - begin);
                    for (size_t i = begin; i < end; ++i) {
                        const std::string& f = files[i];
                        int label = getLabelFromPath(f);
                        if (label < 0) continue;

                        ConvMap input(1, 32, 32);
                        if (!loadImagePreprocessed(f, input)) continue;

                        pushFlat(input, label, out.data, out.labels);
                        out.names.push_back(f);   // 与 pushFlat 成对，保证下标对齐
                    }
                });
            }
            for (auto& wk : loadWorkers) wk.join();

            // 主线程按线程顺序无锁合并（= 原文件顺序），保证 data/labels/names 三者对齐
            size_t totalN = 0;
            for (auto& r : results) totalN += r.labels.size();
            data.resize(totalN * 1024);
            labels.resize(totalN);
            names.resize(totalN);
            size_t off = 0;
            for (auto& r : results) {
                if (r.labels.empty()) continue;
                std::copy(r.data.begin(), r.data.end(), data.begin() + (ptrdiff_t)(off * 1024));
                std::copy(r.labels.begin(), r.labels.end(), labels.begin() + (ptrdiff_t)off);
                std::copy(r.names.begin(), r.names.end(), names.begin() + (ptrdiff_t)off);
                off += r.labels.size();
            }
        }

        const int N = (int)labels.size();
        if (N == 0) { std::cerr << "没有可用测试文件\n"; return 1; }
        if (th > 1)
            std::cout << "CPU test 并行: threads="
                << (int)std::min<size_t>((size_t)th, (size_t)N) << "\n";

        // 单样本前向：从 1024 维输入算 C 类 softmax 概率写入 probOut。
        // 只读全局模型参数（KERNELS_L/BIAS_L/BN_* 与 fc）；调用前已 scatter 好且在
        // 并行期间不变；每次调用独立分配 ConvMap，无共享写 → 多线程安全、数值逐位一致。
        auto forwardOne = [&](const float* src, std::vector<float>& probOut) {
            ConvMap input(1, 32, 32);
            for (int h = 0; h < 32; ++h)
                for (int w = 0; w < 32; ++w)
                    input.data[0][h][w] = src[h * 32 + w];

            ConvMap p2 = forwardConv(input);
            std::vector<float> flat = flatten(p2);
            probOut = softmax(fc.forward(flat));
        };

        // 逐模型逐样本概率张量 allProb[(m*N+i)*C+c]，C = g_net.classes
        std::vector<float> allProb((size_t)N * M * g_net.classes, 0.0f);
        // 前向线程数：下限 1，上限样本数（--threads 1 退化为原单线程）
        const int fwdT = (int)std::min<size_t>((size_t)th, (size_t)N);

        if (!ttaEnabled) {
            for (int m = 0; m < M; ++m) {
                scatterParams(paramsList[m].data(), fc); // 切换到第 m 个模型的全部权重
                if (fwdT <= 1) {
                    // legacy 单线程：与改动前逐位一致
                    for (int i = 0; i < N; ++i) {
                        const float* src = data.data() + (size_t)i * 1024;
                        std::vector<float> prob;
                        forwardOne(src, prob);

                        float* dst = allProb.data() + ((size_t)m * N + i) * g_net.classes;
                        for (int c = 0; c < g_net.classes; ++c) dst[c] = prob[c];
                    }
                }
                else {
                    // 并行：每线程写自己区间内样本的独立 allProb 位置，无共享写；
                    // 模型参数在该模型并行期间只读（scatter 已由主线程完成）。
                    std::vector<std::thread> fwdWorkers;
                    fwdWorkers.reserve((size_t)fwdT);
                    for (int fw = 0; fw < fwdT; ++fw) {
                        const size_t begin = (size_t)N * (size_t)fw / (size_t)fwdT;
                        const size_t end   = (size_t)N * (size_t)(fw + 1) / (size_t)fwdT;
                        fwdWorkers.emplace_back([&, m, begin, end]() {
                            std::vector<float> prob;
                            for (size_t i = begin; i < end; ++i) {
                                const float* src = data.data() + i * 1024;
                                forwardOne(src, prob);

                                float* dst = allProb.data() + ((size_t)m * N + i) * g_net.classes;
                                for (int c = 0; c < g_net.classes; ++c) dst[c] = prob[c];
                            }
                        });
                    }
                    for (auto& wk : fwdWorkers) wk.join();
                }
            }
        }
        else {
            int T = ttaClampK(ttaK);
            std::cout << "TTA: on  k=" << T << "（log 概率空间平均）\n";
            std::vector<double> logitAcc((size_t)N * M * g_net.classes, 0.0);
            std::vector<float> dataT;
            int nc = 0; const TTATransform* cand = ttaCandidates(nc);
            for (int t = 0; t < T; ++t) {
                const std::vector<float>* pData = &data;   // t0 原图零拷贝
                if (cand[t].dx != 0 || cand[t].dy != 0 || cand[t].deg != 0.0f) {
                    makeTTADataForTransform(data, N, t, dataT);
                    pData = &dataT;
                }
                for (int m = 0; m < M; ++m) {
                    scatterParams(paramsList[m].data(), fc);
                    if (fwdT <= 1) {
                        // legacy 单线程：logitAcc 累加顺序仍为 t=0..T-1，逐位一致
                        for (int i = 0; i < N; ++i) {
                            const float* src = pData->data() + (size_t)i * 1024;
                            std::vector<float> prob;
                            forwardOne(src, prob);

                            double* acc = logitAcc.data() + ((size_t)m * N + i) * g_net.classes;
                            for (int c = 0; c < g_net.classes; ++c)
                                acc[c] += std::log((double)prob[c] + 1e-9);
                        }
                    }
                    else {
                        // 并行：每个 (m,i) 的累加器位置独立；t 仍串行，
                        // 故每个样本的概率累加顺序不变 → 逐位一致。
                        std::vector<std::thread> fwdWorkers;
                        fwdWorkers.reserve((size_t)fwdT);
                        for (int fw = 0; fw < fwdT; ++fw) {
                            const size_t begin = (size_t)N * (size_t)fw / (size_t)fwdT;
                            const size_t end   = (size_t)N * (size_t)(fw + 1) / (size_t)fwdT;
                            fwdWorkers.emplace_back([&, m, begin, end]() {
                                std::vector<float> prob;
                                for (size_t i = begin; i < end; ++i) {
                                    const float* src = pData->data() + i * 1024;
                                    forwardOne(src, prob);

                                    double* acc = logitAcc.data() + ((size_t)m * N + i) * g_net.classes;
                                    for (int c = 0; c < g_net.classes; ++c)
                                        acc[c] += std::log((double)prob[c] + 1e-9);
                                }
                            });
                        }
                        for (auto& wk : fwdWorkers) wk.join();
                    }
                }
            }
            logitAccToProb(logitAcc, N, M, T, allProb);
        }

        int correct = 0, total = 0, wrong = 0;
        std::vector<int> pc(g_net.classes, 0), pt(g_net.classes, 0);
        std::ofstream errFile;
        if (outError) {
            errFile.open("outerror.txt");
            std::cout << "\n===== 判错样本清单 (--outerror) =====\n";
            if (errFile) errFile << "===== 判错样本清单 =====\n";
            if (errFile) errFile << "模式: CPU  模型数: " << M
                                 << "  样本数: " << N << "\n";
        }
        for (int i = 0; i < N; ++i) {
            int label = labels[i];
            int pred = fusePredictSample(allProb.data(), N, M, i, w,
                                         fuseMode, fuseMargin, fuseTemp);
            if (pred == label) ++correct;
            ++total;
            if (label >= 0 && label < (int)pt.size()) {
                pt[label]++;
                if (pred == label) pc[label]++;
            }

            if (outError && pred != label) {
                ++wrong;
                // 置信度 = 该样本融合后预测类的加权平均概率
                float conf  = modelWeightedProb(allProb.data(), N, M, i, w, pred);
                float tprob = modelWeightedProb(allProb.data(), N, M, i, w, label);
                std::ostringstream oss;
                oss << "[错] " << names[i]
                    << "  真值=" << label
                    << "  预测=" << pred
                    << "  置信度=" << conf
                    << "  真值概率=" << tprob;
                std::string line = oss.str();
                std::cout << line << "\n";
                if (errFile) errFile << line << "\n";
            }
        }
        if (outError) {
            std::cout << "判错总数: " << wrong << " / " << total << "\n";
            if (errFile) errFile << "判错总数: " << wrong << " / " << total << "\n";
            if (wrong == 0) {
                std::cout << "（未发现判错样本）\n";
                if (errFile) errFile << "（未发现判错样本）\n";
            }
        }

        std::cout << "标准准确率: " << (float)correct / total
            << " (" << correct << "/" << total << ")\n";
        for (int i = 0; i < (int)pt.size(); ++i) {
            if (pt[i] == 0) continue;
            std::cout << "数字 " << i << ": "
                << (float)pc[i] / pt[i]
                << " (" << pc[i] << "/" << pt[i] << ")\n";
        }
        return 0;
    }

    // ========== 默认：单图前向 ==========
    if (!loadModel(modelPath, fc)) return 1;
    std::cout << "已从 " << modelPath << " 加载模型\n";

    ConvMap input(1, 32, 32);
    if (!loadImagePreprocessed(path, input)) return 1;

    ConvMap p2 = forwardConv(input);
    std::vector<float> flat = flatten(p2);
    std::vector<float> prob = softmax(fc.forward(flat));

    int pred = argmaxIndex(prob.data(), (int)prob.size());

    std::cout << "预测: " << pred << "\n";
    for (int i = 0; i < (int)prob.size(); ++i)
        std::cout << i << ": " << prob[i] << "\n";

    return 0;
}
