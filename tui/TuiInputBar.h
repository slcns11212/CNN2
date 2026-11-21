#pragma once

#include <functional>
#include <string>
#include <vector>

#include "TuiCommandCompleter.h"
#include "TuiTheme.h"

namespace cnn2 {
namespace tui {

struct InputBarOptions {
  std::string placeholder = "输入指令…";
  std::string hint = "Enter 发送 · Ctrl+C 退出";
  // P1 暂不渲染发送按钮；字段保留，P5 增强时使用。
  bool show_send_button = false;
  Theme theme = Theme::Default();
};

// 单行输入栏：候选面板 + Tab 循环 + ↑↓ 历史 + Esc 关闭 + 圆角聚焦边框。
// 补全由外部注入的 CommandCompleter 驱动（nullptr 表示关闭补全）。
class InputBar {
 public:
  explicit InputBar(InputBarOptions options = {});

  // 内部 ftxui::Input 保存了 draft_/cursor_ 的裸指针，拷贝/移动都会导致悬垂。
  InputBar(const InputBar&) = delete;
  InputBar& operator=(const InputBar&) = delete;
  InputBar(InputBar&&) = delete;
  InputBar& operator=(InputBar&&) = delete;

  ftxui::Component GetComponent();

  void SetOnSubmit(std::function<void(std::string)> cb);
  // nullptr 关闭补全；非空则立即按当前草稿重算候选。
  void SetCompleter(CommandCompleter* c);
  void ClearInput();
  void Focus();
  const std::string& Text() const;

  // 候选面板（固定高度）+ 圆角输入框（聚焦变色）+ 提示行。
  ftxui::Element Render();

 private:
  void Submit();
  std::string TakeText();
  void RecomputeCandidates();
  void ApplyCandidate(int index);
  bool OnCatchEvent(ftxui::Event event);
  ftxui::Element RenderCompletions() const;

  // 必须最先声明：ftxui::Input 持有 &draft_ 与 &cursor_ 裸指针，
  // 析构顺序要求这些成员晚于 input_ 销毁。
  std::string draft_;
  int cursor_ = 0;
  std::vector<Candidate> candidates_;
  int selected_ = -1;  // -1 表示当前无选中项（首次 Tab 才从第 1 项开始）。
  int token_begin_ = 0;
  int token_end_ = 0;
  bool suppress_recompute_ = false;

  InputBarOptions opt_;
  std::function<void(std::string)> on_submit_;
  CommandCompleter* completer_ = nullptr;

  // 必须最后声明（析构顺序）。
  ftxui::Component input_;
  ftxui::Component container_;
  ftxui::Component root_;
};

}  // namespace tui
}  // namespace cnn2
