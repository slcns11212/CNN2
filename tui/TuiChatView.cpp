// TuiChatView.cpp —— 消息流视图：纯文本折行（保留空格），自动吸附底部。
#include "TuiChatView.h"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cnn2 {
namespace tui {

namespace {

// 历史消息上限：超出后丢弃最旧的消息，避免长会话内存与渲染成本无限增长。
constexpr std::size_t kMaxMessages = 200;

// 单条消息行数上限：流式输出可能无限增长，超过后保留尾部。
constexpr std::size_t kMaxLinesPerMessage = 400;

// 每次滚轮事件滚动的文本行数（按行滚动，而非按文档比例）。
constexpr int kWheelLines = 3;

// 取 UTF-8 首字节对应的码点字节长度；非法/续字节按 1 处理。
std::size_t Utf8GlyphLen(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead >> 5) == 0x06) return 2;
  if ((lead >> 4) == 0x0E) return 3;
  if ((lead >> 3) == 0x1E) return 4;
  return 1;
}

// 按显示宽度折行，逐个 UTF-8 码点累加，保留原始空格（不折叠）。
std::vector<std::string> WrapLine(const std::string& line, int width) {
  std::vector<std::string> out;
  const int limit = std::max(1, width);

  std::string cur;
  int cur_width = 0;
  for (std::size_t i = 0; i < line.size();) {
    std::size_t len = Utf8GlyphLen(static_cast<unsigned char>(line[i]));
    if (i + len > line.size()) {
      len = line.size() - i;
    }
    const std::string glyph = line.substr(i, len);
    const int glyph_width = ftxui::string_width(glyph);

    // 超宽且当前行已有内容时先行换行；单个超宽字符独占一行。
    if (cur_width + glyph_width > limit && !cur.empty()) {
      out.push_back(cur);
      cur.clear();
      cur_width = 0;
    }
    cur += glyph;
    cur_width += glyph_width;
    i += len;
  }
  out.push_back(cur);  // 空行也保留一行
  return out;
}

// 把纯文本按 '\n' 分段后逐行折行，用 vbox 拼成元素。
ftxui::Element RenderPlainText(const std::string& text,
                               int wrap_width,
                               ftxui::Color color) {
  using namespace ftxui;

  Elements lines;
  std::size_t start = 0;
  while (true) {
    const std::size_t nl = text.find('\n', start);
    const std::size_t len =
        (nl == std::string::npos) ? std::string::npos : (nl - start);
    const std::string segment = text.substr(start, len);
    for (const auto& wrapped : WrapLine(segment, wrap_width)) {
      lines.push_back(ftxui::text(wrapped) | ftxui::color(color));
    }
    if (nl == std::string::npos) {
      break;
    }
    start = nl + 1;
  }
  return vbox(std::move(lines));
}

ftxui::Color ColorForRole(const Theme& t, Role r) {
  switch (r) {
    case Role::User:
      return t.accent;
    case Role::Assistant:
      return t.text;
    case Role::System:
      return t.system;
    case Role::Tool:
      return t.success;
  }
  return t.text;
}

// 角色标记直接用 UTF-8 字节写，不依赖源文件编码/执行字符集。
std::string MarkerForRole(Role r) {
  switch (r) {
    case Role::User:
      return "\xe2\x94\x83";  // U+2503 ┃
    case Role::Assistant:
      return "\xe2\x97\x8f";  // U+25CF ●
    case Role::System:
      return "\xc2\xb7";      // U+00B7 ·
    case Role::Tool:
      return "\xe2\x96\xb8";  // U+25B8 ▸
  }
  return "\xc2\xb7";
}

std::string RoleName(Role r) {
  switch (r) {
    case Role::User:
      return "YOU";
    case Role::Assistant:
      return "ASSISTANT";
    case Role::System:
      return "SYSTEM";
    case Role::Tool:
      return "OUTPUT";  // Tool 显示名
  }
  return "SYSTEM";
}

// 超出行数上限时从开头丢弃多余行，保留尾部内容。
void TrimToMaxLines(std::string& text) {
  std::size_t newlines = 0;
  for (char c : text) {
    if (c == '\n') ++newlines;
  }
  if (newlines + 1 <= kMaxLinesPerMessage) {
    return;
  }
  std::size_t drop = newlines + 1 - kMaxLinesPerMessage;
  std::size_t pos = 0;
  while (drop > 0) {
    const std::size_t p = text.find('\n', pos);
    if (p == std::string::npos) {
      break;
    }
    pos = p + 1;
    --drop;
  }
  text.erase(0, pos);
}

}  // namespace

ChatView::ChatView() : theme_(Theme::Default()) {}

void ChatView::AddMessage(Role role, std::string text, std::string timestamp) {
  Message m;
  m.role = role;
  TrimToMaxLines(text);
  m.text = std::move(text);
  m.timestamp = std::move(timestamp);

  std::lock_guard<std::mutex> lock(mutex_);
  messages_.push_back(std::move(m));
  if (messages_.size() > kMaxMessages) {
    const std::size_t excess = messages_.size() - kMaxMessages;
    messages_.erase(messages_.begin(),
                    messages_.begin() + static_cast<std::ptrdiff_t>(excess));
    // 丢弃旧消息后缓存索引整体错位，只能整体重建。
    cache_.clear();
    cache_wrap_ = -1;
    dirty_index_ = -1;
  }
  if (follow_bottom_) {
    scroll_line_ = 0;
  }
}

void ChatView::AppendToLast(std::string delta) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (messages_.empty()) {
    return;
  }
  messages_.back().text += delta;
  TrimToMaxLines(messages_.back().text);
  // 只标记被追加的消息为脏：Render 时仅重建该项，不整体重建。
  dirty_index_ = static_cast<int>(messages_.size() - 1);
  if (follow_bottom_) {
    scroll_line_ = 0;
  }
}

void ChatView::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  messages_.clear();
  cache_.clear();
  cache_wrap_ = -1;
  dirty_index_ = -1;
  follow_bottom_ = true;
  scroll_line_ = 0;
}

void ChatView::SetWrapWidth(int cols) {
  const int clamped = std::clamp(cols, 10, 200);
  std::lock_guard<std::mutex> lock(mutex_);
  if (clamped == wrap_width_) {
    return;
  }
  wrap_width_ = clamped;
  cache_.clear();
  cache_wrap_ = -1;
  dirty_index_ = -1;
}

int ChatView::WrapWidth() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return wrap_width_;
}

void ChatView::SetTheme(const Theme& theme) {
  std::lock_guard<std::mutex> lock(mutex_);
  theme_ = theme;
  cache_.clear();
  cache_wrap_ = -1;
  dirty_index_ = -1;
}

std::size_t ChatView::Count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return messages_.size();
}

ftxui::Element ChatView::RenderMessage(const Message& m) const {
  using namespace ftxui;

  const Color role_color = ColorForRole(theme_, m.role);

  // 正文缩进 4 列，纯文本折行（paragraph 会折叠连续空格，这里不能用）。
  Element body = hbox({
      ftxui::text("    "),
      RenderPlainText(m.text, wrap_width_, theme_.text) | flex,
  });

  return vbox({
      hbox({
          ftxui::text(MarkerForRole(m.role)) | bold | color(role_color),
          ftxui::text(" "),
          ftxui::text(RoleName(m.role)) | bold | color(role_color),
          filler(),
          m.timestamp.empty() ? ftxui::text("")
                              : (ftxui::text(m.timestamp) | color(theme_.muted)),
      }),
      std::move(body),
      ftxui::text(" "),  // 消息之间空 1 行
  });
}

ftxui::Element ChatView::Render() {
  using namespace ftxui;

  std::lock_guard<std::mutex> lock(mutex_);

  // 折行宽度与视图宽度联动：终端尺寸变化后 box_（上一帧 reflect 结果）
  // 会变，这里按「视图宽度 - 4 列缩进 - 1 列滚动条」重设正文折行宽度。
  // 首帧 box_ 尚未填充（宽度为 1），用 >= 20 的阈值跳过。
  if (box_.x_max >= box_.x_min) {
    const int avail = box_.x_max - box_.x_min + 1;
    if (avail >= 20) {
      const int usable = std::clamp(avail - 5, 10, 200);
      if (usable != wrap_width_) {
        wrap_width_ = usable;
        cache_.clear();
        cache_wrap_ = -1;
        dirty_index_ = -1;
      }
    }
  }

  // 缓存维护策略：
  // - 折行宽度/主题变化 -> 全量重建（cache_wrap_ 置 -1）；
  // - 消息数增加 -> 只补建新增消息；
  // - 消息数减少 -> 截断；
  // - 流式追加 -> 只重建 dirty_index_ 指向的消息。
  if (cache_wrap_ != wrap_width_) {
    cache_.clear();
    cache_.reserve(messages_.size());
    for (const auto& m : messages_) {
      cache_.push_back(RenderMessage(m));
    }
    cache_wrap_ = wrap_width_;
    dirty_index_ = -1;
  } else {
    if (cache_.size() < messages_.size()) {
      cache_.reserve(messages_.size());
      for (std::size_t i = cache_.size(); i < messages_.size(); ++i) {
        cache_.push_back(RenderMessage(messages_[i]));
      }
    } else if (cache_.size() > messages_.size()) {
      cache_.resize(messages_.size());
    }
    if (dirty_index_ >= 0 &&
        static_cast<std::size_t>(dirty_index_) < cache_.size()) {
      const std::size_t idx = static_cast<std::size_t>(dirty_index_);
      cache_[idx] = RenderMessage(messages_[idx]);
    }
    dirty_index_ = -1;
  }

  if (cache_.empty()) {
    return vbox({ftxui::text(" ")}) | focusPosition(0, 0) |
           vscroll_indicator | yframe | flex | reflect(box_);
  }

  // 按文档总行数换算焦点位置，实现“按行”滚动：
  // scroll_line_ 为距底部的行数，y 为文档内焦点行号。
  Element doc = vbox(cache_);
  doc->ComputeRequirement();
  const int total = std::max(1, doc->requirement().min_y);
  const int y = std::clamp(total - scroll_line_, 0, total);
  // vscroll_indicator 必须在 yframe 内侧，才能正确绘制滚动条。
  return doc | focusPosition(0, y) | vscroll_indicator | yframe | flex |
         reflect(box_);
}

ftxui::Component ChatView::GetComponent() {
  if (!component_) {
    auto base = ftxui::Renderer([this] { return Render(); });
    // 滚轮按行上下滚动消息区；仅处理落在消息区内的滚轮事件。
    component_ = ftxui::CatchEvent(base, [this](ftxui::Event event) {
      if (!event.is_mouse()) {
        return false;
      }
      const ftxui::Mouse& mouse = event.mouse();
      if (!box_.Contain(mouse.x, mouse.y)) {
        return false;
      }
      const ftxui::Mouse::Button button = mouse.button;
      if (button != ftxui::Mouse::WheelUp &&
          button != ftxui::Mouse::WheelDown) {
        return false;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (button == ftxui::Mouse::WheelUp) {
        follow_bottom_ = false;
        scroll_line_ += kWheelLines;
      } else {
        scroll_line_ = std::max(0, scroll_line_ - kWheelLines);
        if (scroll_line_ == 0) {
          follow_bottom_ = true;
        }
      }
      return true;
    });
  }
  return component_;
}

}  // namespace tui
}  // namespace cnn2
