#pragma once

#include <string>
#include <vector>

namespace cnn2 {
namespace tui {

struct Candidate {
    std::string label;    // 显示文本
    std::string insert;   // 插入文本（值候选可含尾随空格）
    std::string hint;     // 右侧说明（可空）
};

struct CompleterResult {
    std::vector<Candidate> candidates;
    int begin = 0;   // 当前 token 起始字节下标
    int end = 0;     // 当前 token 结束字节下标（<= cursor）
};

class CommandCompleter {
 public:
    CommandCompleter();
    void SetKnownKeys(const char* const* keys, int count);

    // draft/cursor 均为字节下标；返回当前 token 的候选与区间
    CompleterResult Complete(const std::string& draft, int cursor) const;

    bool HistoryPrev(std::string& draft);
    bool HistoryNext(std::string& draft);
    void AddHistory(const std::string& line);
    void ClearHistory();
    const std::vector<std::string>& History() const { return history_; }

 private:
    std::vector<std::string> known_keys_;   // 来自 main.cpp knownArgKeys()（不含 --）
    std::vector<std::string> history_;
    int history_pos_ = -1;
    std::string stash_;                     // 首次按↑前的草稿
};

}  // namespace tui
}  // namespace cnn2
