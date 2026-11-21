// TuiCommandCompleter.cpp —— TUI 命令补全核心（纯 C++，不依赖 FTXUI / CNN2 内部头）。
//
// 职责：
//   1) 按当前草稿与光标位置计算 token 区间，产出命令 / 参数键 / 枚举值候选；
//   2) 维护一份命令历史，提供 ↑/↓ 移动接口。
// 线程模型：P5a 假设由 UI 线程单线程调用，内部不加锁。
#include "TuiCommandCompleter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cnn2 {
namespace tui {

namespace {

// 键名规范形式：与 main.cpp canonicalArgKey 同规则。
// '-' -> '_'，其余转小写。用于大小写不敏感 + 下划线/连字符等价匹配。
std::string Canon(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c == '-') {
            c = '_';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

// 去掉前导 '-'（参数键可能以 -- 开头）后再做 Canon。
std::string CanonKey(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && s[i] == '-') {
        ++i;
    }
    return Canon(s.substr(i));
}

bool StartsWith(const std::string& item, const std::string& prefix) {
    if (prefix.size() > item.size()) {
        return false;
    }
    return item.compare(0, prefix.size(), prefix) == 0;
}

std::string Trim(const std::string& s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

bool IsSpace(char c) {
    return c == ' ' || c == '\t';
}

// 一个 token 的字节区间与文本。
struct Span {
    int begin = 0;
    int end = 0;
    std::string text;
    bool trailing_space = false;
};

// 按空格/Tab 切分。字符串末尾为空白（或字符串为空）时追加一个空 token，
// 作为「空格触发新一轮推测」的锚点。
std::vector<Span> SplitTokens(const std::string& s) {
    std::vector<Span> out;
    const int n = static_cast<int>(s.size());
    int i = 0;
    while (i < n) {
        while (i < n && IsSpace(s[static_cast<std::size_t>(i)])) {
            ++i;
        }
        if (i >= n) {
            break;
        }
        const int b = i;
        while (i < n && !IsSpace(s[static_cast<std::size_t>(i)])) {
            ++i;
        }
        Span sp;
        sp.begin = b;
        sp.end = i;
        sp.text = s.substr(static_cast<std::size_t>(b),
                           static_cast<std::size_t>(i - b));
        sp.trailing_space = (i < n);
        out.push_back(std::move(sp));
    }
    if (out.empty() || s.empty() || IsSpace(s.back())) {
        Span sp;
        sp.begin = n;
        sp.end = n;
        out.push_back(std::move(sp));
    }
    return out;
}

// 首 token 命令词表。
const std::vector<std::string>& CommandWords() {
    static const std::vector<std::string> kWords = {
        "gen",    "train", "bkc",   "test", "infer",   "read",
        "gui",    "checkdata", "gradcheck", "version", "help", "clear",
        "quit",   "cd",    "history", "status",
    };
    return kWords;
}

std::string HintForCommand(const std::string& cmd) {
    static const std::map<std::string, std::string> kHints = {
        {"gen", "生成模型"},
        {"train", "训练模型"},
        {"bkc", "训练（bkc 别名）"},
        {"test", "测试/集成评估"},
        {"infer", "单图推理"},
        {"read", "读取数据概览"},
        {"gui", "启动图形界面"},
        {"checkdata", "校验数据集"},
        {"gradcheck", "梯度检查"},
        {"version", "显示版本"},
        {"help", "帮助"},
        {"clear", "清空界面"},
        {"quit", "退出 TUI"},
        {"cd", "切换工作目录"},
        {"history", "显示命令历史"},
        {"status", "显示状态"},
    };
    const auto it = kHints.find(cmd);
    return it == kHints.end() ? std::string() : it->second;
}

// 固定取值键（键名用 Canon 后的形式比较）。
std::vector<std::string> ValuesFor(const std::string& canon_key) {
    if (canon_key == "fuse") {
        return {"soft", "logit", "geom", "vote", "gate", "wbest"};
    }
    if (canon_key == "mix") {
        return {"balanced", "full"};
    }
    if (canon_key == "prep") {
        return {"crop", "deskew", "resize", "center", "thick", "none"};
    }
    return {};
}

std::string HintForValue(const std::string& canon_key, const std::string& value) {
    if (canon_key == "fuse") {
        if (value == "soft") return "softmax 融合";
        if (value == "logit") return "logit 融合";
        if (value == "geom") return "几何平均";
        if (value == "vote") return "投票";
        if (value == "gate") return "门控融合";
        if (value == "wbest") return "最佳权重";
    } else if (canon_key == "mix") {
        if (value == "balanced") return "类别均衡采样";
        if (value == "full") return "全量采样";
    } else if (canon_key == "prep") {
        if (value == "crop") return "裁剪";
        if (value == "deskew") return "纠斜";
        if (value == "resize") return "缩放";
        if (value == "center") return "居中";
        if (value == "thick") return "加粗";
        if (value == "none") return "不做预处理";
    }
    return std::string();
}

// 各模式可用键（原始键名）。
const std::vector<std::string>& TrainKeys() {
    static const std::vector<std::string> kKeys = {
        "epochs", "lr_fc", "lr_conv1", "lr_conv2", "momentum", "nesterov",
        "model", "batch", "gpu", "cpu", "time", "log", "val", "strict-data",
        "threads", "outerror", "seed", "data2", "mix", "full", "wd", "epsave",
        "epsn", "txtmodel", "P", "BN", "res", "res2", "preact", "c1", "c2",
        "depth", "no_conv1", "no_conv2", "no_fc", "onecycle", "lr_warmup",
        "lr_min", "lr_peak", "aug", "aug_k", "aug_shift", "aug_rotate",
        "aug_scale", "aug_elastic", "aug_elastic_sigma", "aug_thick", "prep",
        "prep_target", "prep_stretch", "prep_ink", "prep_deskew_max", "diag",
        "fuse", "fuse_margin", "fuse_temp", "weights", "tta", "tta_k",
        "fpng",
    };
    return kKeys;
}

const std::vector<std::string>& TestKeys() {
    static const std::vector<std::string> kKeys = {
        "model", "fuse", "weights", "fuse_margin", "fuse_temp", "tta",
        "tta_k", "outerror", "batch", "gpu", "cpu", "prep", "prep_target",
        "prep_stretch", "prep_ink", "prep_deskew_max", "diag",
    };
    return kKeys;
}

const std::vector<std::string>& GenKeys() {
    static const std::vector<std::string> kKeys = {
        "model", "out", "depth", "c1", "c2", "BN", "P", "res", "res2",
        "preact", "no_conv1", "no_conv2", "no_fc", "txtmodel", "seed",
    };
    return kKeys;
}

const std::vector<std::string>& InferKeys() {
    static const std::vector<std::string> kKeys = {
        "model", "cpu", "gpu", "prep", "prep_target", "prep_stretch",
        "prep_ink", "prep_deskew_max", "diag",
    };
    return kKeys;
}

// 按首 token 给出的模式返回可用键；未知模式回退全量 known_keys_。
std::vector<std::string> KeysForMode(const std::string& mode,
                                     const std::vector<std::string>& known) {
    const std::vector<std::string>* desired = nullptr;
    if (mode == "bkc" || mode == "train") {
        desired = &TrainKeys();
    } else if (mode == "test") {
        desired = &TestKeys();
    } else if (mode == "gen") {
        desired = &GenKeys();
    } else if (mode == "infer") {
        desired = &InferKeys();
    }
    if (desired == nullptr) {
        return known;
    }
    if (known.empty()) {
        return *desired;
    }
    // 与 known 求交，保持模式表的顺序；返回 known 中的原始键名。
    std::vector<std::string> out;
    for (const std::string& k : *desired) {
        const std::string ck = Canon(k);
        for (const std::string& kk : known) {
            if (Canon(kk) == ck) {
                out.push_back(kk);
                break;
            }
        }
    }
    return out;
}

std::string HintForKey(const std::string& raw_key) {
    static const std::map<std::string, std::string> kHints = {
        {"epochs", "训练轮数"},
        {"lr_fc", "全连接层学习率"},
        {"lr_conv1", "卷积层1学习率"},
        {"lr_conv2", "卷积层2学习率"},
        {"momentum", "动量"},
        {"nesterov", "Nesterov 动量"},
        {"model", "模型文件"},
        {"out", "输出模型文件"},
        {"batch", "批大小"},
        {"gpu", "启用 GPU"},
        {"cpu", "强制 CPU"},
        {"time", "训练时长限制"},
        {"log", "日志文件"},
        {"val", "验证集"},
        {"strict_data", "严格数据校验"},
        {"threads", "线程数"},
        {"outerror", "输出错误样本"},
        {"seed", "随机种子"},
        {"data2", "第二数据目录"},
        {"mix", "采样方式"},
        {"full", "全量数据"},
        {"wd", "权重衰减"},
        {"epsave", "保存轮数间隔"},
        {"epsn", "结束轮数"},
        {"txtmodel", "文本模型文件"},
        {"p", "激活/损失类型"},
        {"bn", "BatchNorm"},
        {"res", "残差连接"},
        {"res2", "二级残差"},
        {"preact", "预激活"},
        {"c1", "通道数1"},
        {"c2", "通道数2"},
        {"depth", "深度"},
        {"no_conv1", "禁用卷积1"},
        {"no_conv2", "禁用卷积2"},
        {"no_fc", "禁用全连接"},
        {"onecycle", "One-Cycle 调度"},
        {"lr_warmup", "学习率预热"},
        {"lr_min", "最小学习率"},
        {"lr_peak", "峰值学习率"},
        {"fuse", "融合方式"},
        {"fuse_margin", "融合边界"},
        {"fuse_temp", "融合温度"},
        {"weights", "权重文件列表"},
        {"tta", "TTA 次数"},
        {"tta_k", "TTA 参数"},
        {"prep", "预处理方式"},
        {"prep_target", "预处理目标"},
        {"prep_stretch", "预处理拉伸"},
        {"prep_ink", "墨迹处理"},
        {"prep_deskew_max", "最大纠斜角"},
        {"diag", "诊断输出"},
        {"aug", "数据增强"},
        {"aug_k", "增强强度"},
        {"aug_shift", "增强平移"},
        {"aug_rotate", "增强旋转"},
        {"aug_scale", "增强缩放"},
        {"aug_elastic", "弹性增强"},
        {"aug_elastic_sigma", "弹性增强 sigma"},
        {"aug_thick", "增强加粗"},
        {"gc_samples", "梯度检查样本数"},
        {"gc_eps", "梯度检查 epsilon"},
        {"fpng", "快速 PNG"},
        {"gui", "图形界面"},
        {"tui", "启用 TUI"},
        {"version", "显示版本"},
        {"gradcheck", "梯度检查"},
        {"checkdata", "校验数据"},
        {"cnn2", "内部模式"},
        {"help", "帮助"},
        {"read", "读取数据"},
    };
    const auto it = kHints.find(CanonKey(raw_key));
    return it == kHints.end() ? std::string() : it->second;
}

}  // namespace

CommandCompleter::CommandCompleter() = default;

void CommandCompleter::SetKnownKeys(const char* const* keys, int count) {
    known_keys_.clear();
    if (keys == nullptr || count <= 0) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        if (keys[i] != nullptr) {
            known_keys_.push_back(keys[i]);
        }
    }
}

CompleterResult CommandCompleter::Complete(const std::string& draft,
                                           int cursor) const {
    CompleterResult out;
    const int size = static_cast<int>(draft.size());
    const int cur = std::min(std::max(cursor, 0), size);
    const std::vector<Span> toks = SplitTokens(draft);

    // 按光标位置定位当前 token（而不是固定取最后一个 token）：
    //   1) 优先匹配包含光标的有内容 token（cur ∈ [begin, end]）；
    //   2) 其次匹配光标处的空锚点 token（draft 以空白结尾时 SplitTokens
    //      会补一个 begin == end == size 的空 token）；
    //   3) 最后合成一个空 span（光标落在两个 token 之间的空白里）。
    int curIdx = -1;
    Span curSpan{cur, cur, "", true};
    for (int i = 0; i < static_cast<int>(toks.size()); ++i) {
        const Span& sp = toks[static_cast<std::size_t>(i)];
        if (sp.end <= sp.begin) {
            continue;
        }
        if (cur >= sp.begin && cur <= sp.end) {
            curSpan = sp;
            curIdx = i;
            break;
        }
    }
    if (curIdx < 0) {
        for (int i = 0; i < static_cast<int>(toks.size()); ++i) {
            const Span& sp = toks[static_cast<std::size_t>(i)];
            if (sp.end == sp.begin && sp.begin == cur) {
                curSpan = sp;
                curIdx = i;
                break;
            }
        }
    }
    if (curIdx < 0) {
        curSpan = Span{cur, cur, "", true};
        for (int i = 0; i < static_cast<int>(toks.size()); ++i) {
            const Span& sp = toks[static_cast<std::size_t>(i)];
            if (sp.end <= cur) {
                curIdx = i;
            } else {
                break;
            }
        }
    }

    // 替换区间只覆盖光标之前的部分，硬保证 begin <= end <= cursor。
    const int begin = std::min(std::max(curSpan.begin, 0), cur);
    const int end = std::max(begin, std::min(cur, curSpan.end));
    out.begin = begin;
    out.end = end;
    const std::string curText =
        draft.substr(static_cast<std::size_t>(begin),
                     static_cast<std::size_t>(end - begin));

    // 光标之前的最后一个非空 token（值补全的前一个 token）。
    int prevIdx = -1;
    for (int i = 0; i < static_cast<int>(toks.size()); ++i) {
        const Span& sp = toks[static_cast<std::size_t>(i)];
        if (sp.end > sp.begin && sp.end <= begin) {
            prevIdx = i;
        }
    }
    const bool hasPrev = (prevIdx >= 0);

    // 模式词：草稿第一个非空 token（通常在光标之前，即命令词）。
    std::string modeText;
    for (const Span& sp : toks) {
        if (sp.end > sp.begin) {
            if (sp.end <= begin) {
                modeText = sp.text;
            }
            break;
        }
    }

    // 分支1：值补全 —— 前一个 token 是已知枚举键。
    if (hasPrev) {
        const std::string& prev = toks[static_cast<std::size_t>(prevIdx)].text;
        const std::string prev_canon = CanonKey(prev);
        const std::vector<std::string> values = ValuesFor(prev_canon);
        if (!values.empty()) {
            // 形如 --prep crop,re：只补最后一个逗号段，保留前面部分。
            std::string base;
            std::string prefix = curText;
            const std::size_t comma = curText.rfind(',');
            if (comma != std::string::npos) {
                base = curText.substr(0, comma + 1);
                prefix = curText.substr(comma + 1);
            }
            const std::string lower_prefix = Canon(prefix);
            for (const std::string& v : values) {
                if (!StartsWith(v, lower_prefix)) {
                    continue;
                }
                Candidate c;
                c.label = v;
                c.insert = base + v + " ";
                c.hint = HintForValue(prev_canon, v);
                out.candidates.push_back(std::move(c));
            }
            return out;
        }
    }

    // 分支2：参数键补全 —— 当前 token 以 '-' 开头，或为空且光标前已有内容。
    const bool want_key =
        (!curText.empty() && curText[0] == '-') ||
        (curText.empty() && hasPrev);
    if (want_key) {
        // 前缀匹配使用「当前 token 从 begin 到 token 末尾」的文本：
        // 光标停在 token 中间时（如 --pre|p）仍能给出与该 token 相关的键。
        std::string match_text;
        if (curSpan.end > begin) {
            match_text = draft.substr(
                static_cast<std::size_t>(begin),
                static_cast<std::size_t>(curSpan.end - begin));
        }
        if (match_text.empty()) {
            match_text = curText;
        }

        std::size_t dash = 0;
        while (dash < match_text.size() && match_text[dash] == '-') {
            ++dash;
        }
        const std::string prefix = Canon(match_text.substr(dash));
        const std::vector<std::string> keys =
            KeysForMode(Canon(modeText), known_keys_);
        std::vector<std::string> seen;
        for (const std::string& k : keys) {
            const std::string ck = Canon(k);
            if (!StartsWith(ck, prefix)) {
                continue;
            }
            // 规范形式去重（"BN"/"bn" 等价）。
            bool dup = false;
            for (const std::string& s : seen) {
                if (s == ck) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen.push_back(ck);
            Candidate c;
            c.label = "--" + k;
            c.insert = "--" + k;
            c.hint = HintForKey(k);
            out.candidates.push_back(std::move(c));
        }
        return out;
    }

    // 分支3：首 token → 命令词补全（光标前没有非空 token）。
    if (!hasPrev) {
        for (const std::string& cmd : CommandWords()) {
            if (!StartsWith(cmd, curText)) {
                continue;
            }
            Candidate c;
            c.label = cmd;
            c.insert = cmd;
            c.hint = HintForCommand(cmd);
            out.candidates.push_back(std::move(c));
        }
        return out;
    }

    // 分支4：路径等自由文本位置，不产出候选。
    return out;
}

bool CommandCompleter::HistoryPrev(std::string& draft) {
    if (history_.empty()) {
        return false;
    }
    if (history_pos_ == -1) {
        stash_ = draft;
        history_pos_ = static_cast<int>(history_.size()) - 1;
    } else if (history_pos_ > 0) {
        --history_pos_;
    }
    // 到达最早一条后停住，继续按 ↑ 仍返回该条。
    draft = history_[static_cast<std::size_t>(history_pos_)];
    return true;
}

bool CommandCompleter::HistoryNext(std::string& draft) {
    if (history_pos_ == -1) {
        return false;
    }
    ++history_pos_;
    if (history_pos_ >= static_cast<int>(history_.size())) {
        history_pos_ = -1;
        draft = stash_;
        return true;
    }
    draft = history_[static_cast<std::size_t>(history_pos_)];
    return true;
}

void CommandCompleter::AddHistory(const std::string& line) {
    const std::string t = Trim(line);
    if (t.empty()) {
        return;
    }
    if (!history_.empty() && history_.back() == t) {
        history_pos_ = -1;
        return;
    }
    history_.push_back(t);
    history_pos_ = -1;
}

void CommandCompleter::ClearHistory() {
    history_.clear();
    history_pos_ = -1;
    stash_.clear();
}

}  // namespace tui
}  // namespace cnn2
