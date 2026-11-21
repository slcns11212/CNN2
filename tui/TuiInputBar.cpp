// TuiInputBar.cpp —— 单行输入栏实现：候选面板 + Tab 循环 + ↑↓ 历史。
#include "TuiInputBar.h"

#include <algorithm>
#include <utility>

namespace cnn2 {
namespace tui {

namespace {

// 候选面板最多同时显示的候选行数（不含底部提示行）。
constexpr int kMaxVisibleCandidates = 8;

}  // namespace

InputBar::InputBar(InputBarOptions options) : opt_(std::move(options)) {
  using namespace ftxui;

  // 逐字段赋值（C++17 不支持 designated initializer）。
  InputOption io;
  io.placeholder = opt_.placeholder;
  io.multiline = false;  // 必须 false，否则回车会插入 '\n'。
  io.on_enter = [this] { Submit(); };
  // 让输入组件的内部光标位置与 cursor_ 双向同步（字节下标）。
  io.cursor_position = Ref<int>(&cursor_);

  // 文本自带前景色：避免外层给边框上色时把文字/placeholder 一起染成近黑色。
  io.transform = [theme = opt_.theme](InputState s) {
    Element e = s.element;
    e = e | color(s.is_placeholder ? theme.placeholder : theme.text);
    return e;
  };

  // 单行输入：粘贴进来的 '\r'/'\n' 必须从 draft_ 中剔除，
  // 否则会残留换行（并且在某些终端上一次粘贴触发多次提交）。
  // 内容变化后按当前光标重算候选；补全插入走 ApplyCandidate，
  // 用 suppress_recompute_ 避免直接把面板清掉。
  io.on_change = [this] {
    draft_.erase(std::remove_if(draft_.begin(), draft_.end(),
                                [](char c) { return c == '\r' || c == '\n'; }),
                 draft_.end());
    if (!suppress_recompute_) {
      RecomputeCandidates();
    }
  };

  // CatchEvent 位于 Input 外层：Tab/Esc/↑/↓ 先在这里拦截，
  // 未消费的事件再交给 Input（字符输入、左右移动、退格等）。
  input_ = CatchEvent(Input(&draft_, io),
                      [this](Event e) { return OnCatchEvent(std::move(e)); });

  // P1 不渲染发送按钮（show_send_button 字段保留给后续阶段）。
  container_ = Container::Horizontal({input_});
  root_ = Renderer(container_, [this] { return Render(); });
}

ftxui::Component InputBar::GetComponent() { return root_; }

void InputBar::SetOnSubmit(std::function<void(std::string)> cb) {
  on_submit_ = std::move(cb);
}

void InputBar::SetCompleter(CommandCompleter* c) {
  completer_ = c;
  candidates_.clear();
  selected_ = -1;
  // nullptr 时 RecomputeCandidates 只清空候选；非空时立刻给出首轮候选。
  RecomputeCandidates();
}

void InputBar::ClearInput() {
  draft_.clear();
  cursor_ = 0;
  candidates_.clear();
  selected_ = -1;
  token_begin_ = 0;
  token_end_ = 0;
}

void InputBar::Focus() {
  if (input_) {
    input_->TakeFocus();
  }
}

const std::string& InputBar::Text() const { return draft_; }

std::string InputBar::TakeText() {
  std::string t = std::move(draft_);
  draft_.clear();
  cursor_ = 0;
  return t;
}

void InputBar::RecomputeCandidates() {
  candidates_.clear();
  selected_ = -1;  // 候选重算后回到未选中状态。
  if (completer_ == nullptr) {
    return;
  }
  const int cur =
      std::min<int>(std::max(cursor_, 0), static_cast<int>(draft_.size()));
  CompleterResult r = completer_->Complete(draft_, cur);
  candidates_ = std::move(r.candidates);
  token_begin_ = r.begin;
  token_end_ = r.end;
}

void InputBar::ApplyCandidate(int index) {
  if (index < 0 || index >= static_cast<int>(candidates_.size())) {
    return;
  }
  const std::string& ins = candidates_[static_cast<std::size_t>(index)].insert;

  // 防御：Completer 给出的区间理论上合法，这里仍夹在草稿范围内。
  const int size = static_cast<int>(draft_.size());
  const int b = std::min<int>(std::max(token_begin_, 0), size);
  const int e = std::min<int>(std::max(token_end_, b), size);

  suppress_recompute_ = true;
  draft_.replace(static_cast<std::size_t>(b), static_cast<std::size_t>(e - b),
                 ins);
  cursor_ = b + static_cast<int>(ins.size());
  token_end_ = cursor_;
  suppress_recompute_ = false;
}

bool InputBar::OnCatchEvent(ftxui::Event event) {
  using namespace ftxui;

  // Tab / Shift+Tab：候选循环。必须返回 true，阻止 Container 切换焦点。
  if (event == Event::Tab || event == Event::TabReverse) {
    if (candidates_.empty()) {
      RecomputeCandidates();
    }
    if (!candidates_.empty()) {
      const int n = static_cast<int>(candidates_.size());
      const int dir = (event == Event::Tab) ? 1 : -1;
      // selected_ == -1：首次 Tab 落到第 1 项，首次 Shift+Tab 落到最后一项。
      if (selected_ < 0) {
        selected_ = (dir > 0) ? 0 : (n - 1);
      } else {
        selected_ = (selected_ + dir + n) % n;
      }
      ApplyCandidate(selected_);
    }
    return true;
  }

  // Esc：关闭候选面板。
  if (event == Event::Escape && !candidates_.empty()) {
    candidates_.clear();
    selected_ = -1;
    return true;
  }

  // ↑ / ↓：历史回溯。必须在 Input 之前拦截：
  // Input 在单行且 cursor>0 时会消费方向键。
  if (event == Event::ArrowUp) {
    if (completer_ != nullptr && completer_->HistoryPrev(draft_)) {
      cursor_ = static_cast<int>(draft_.size());
      RecomputeCandidates();
      return true;
    }
    return false;
  }
  if (event == Event::ArrowDown) {
    if (completer_ != nullptr && completer_->HistoryNext(draft_)) {
      cursor_ = static_cast<int>(draft_.size());
      RecomputeCandidates();
      return true;
    }
    return false;
  }

  // 其余事件交给 Input。
  return false;
}

ftxui::Element InputBar::RenderCompletions() const {
  using namespace ftxui;

  const int total = static_cast<int>(candidates_.size());

  // 无候选：只渲染 1 行提示，不再补 8 个空行，避免小终端被固定面板挤压。
  if (total == 0) {
    return hbox({
               text(" Tab 补全 · 输入命令查看候选") | color(opt_.theme.muted),
               filler(),
           }) |
           size(HEIGHT, EQUAL, 1) | bgcolor(opt_.theme.panel);
  }

  const int visible = std::min(total, kMaxVisibleCandidates);

  // 以 selected_ 为中心滚动窗口；selected_ == -1 时窗口从头开始。
  int start = 0;
  if (selected_ >= 0 && total > kMaxVisibleCandidates) {
    start = selected_ - kMaxVisibleCandidates / 2;
    start = std::max(0, std::min(start, total - kMaxVisibleCandidates));
  }

  Elements rows;
  for (int i = 0; i < visible; ++i) {
    const int idx = start + i;
    const Candidate& cand = candidates_[static_cast<std::size_t>(idx)];
    const bool active = (idx == selected_);

    Element label =
        text((active ? "> " : "  ") + cand.label) |
        color(active ? opt_.theme.background : opt_.theme.text);
    Element hint = text(cand.hint) |
                   color(active ? opt_.theme.background : opt_.theme.muted);

    Element line = hbox({std::move(label), filler(), std::move(hint)}) | xflex;
    if (active) {
      line = std::move(line) | bgcolor(opt_.theme.accent) | bold;
    }
    rows.push_back(std::move(line) | size(HEIGHT, EQUAL, 1));
  }

  // 动态高度：候选行数 + 1 行 footer（不再补空行）。
  const std::string footer =
      std::to_string(total) + " 项 · Tab 循环 · Esc 关闭";
  rows.push_back(text(footer) | color(opt_.theme.muted) |
                 size(HEIGHT, EQUAL, 1));

  return vbox(std::move(rows)) | size(HEIGHT, EQUAL, visible + 1) |
         bgcolor(opt_.theme.panel);
}

ftxui::Element InputBar::Render() {
  using namespace ftxui;

  // 用原始 UTF-8 字节写 "❯ "，不依赖源文件编码/执行字符集。
  Element prompt = text(" \xe2\x9d\xaf ") | color(opt_.theme.accent);

  // borderStyled(ROUNDED, Color) 只给边框上色，不递归覆盖文字前景色；
  // 文字颜色由 Input 的 transform 自行设置。
  const Color border_color =
      input_->Focused() ? opt_.theme.border_focus : opt_.theme.border;
  Element input_box = input_->Render() |
                      borderStyled(BorderStyle::ROUNDED, border_color) |
                      bgcolor(opt_.theme.panel) | flex;

  Element row = hbox({prompt, input_box});

  Element hint =
      text(opt_.hint) | color(opt_.theme.muted) | size(HEIGHT, EQUAL, 1);

  return vbox({RenderCompletions(), row, hint});
}

void InputBar::Submit() {
  std::string t = TakeText();
  // 提交后收起候选面板并复位光标。
  candidates_.clear();
  selected_ = -1;
  cursor_ = 0;
  if (!t.empty() && on_submit_) {
    on_submit_(std::move(t));
  }
  // 保证提交后焦点仍在输入框。
  Focus();
}

}  // namespace tui
}  // namespace cnn2
