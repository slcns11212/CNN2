#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include "TuiFtxui.h"
#include "TuiMessage.h"
#include "TuiTheme.h"

namespace cnn2 {
namespace tui {

// 消息流视图：纯文本折行（保留连续空格，不折叠），自动吸附底部。
// 线程安全：AddMessage/AppendToLast/Count 等可在后台线程调用。
class ChatView {
 public:
  ChatView();

  // 内部 ftxui::Component 回调捕获了 this，拷贝/移动会导致悬垂指针。
  ChatView(const ChatView&) = delete;
  ChatView& operator=(const ChatView&) = delete;
  ChatView(ChatView&&) = delete;
  ChatView& operator=(ChatView&&) = delete;

  ftxui::Component GetComponent();
  ftxui::Element Render();

  // 追加一条消息；timestamp 为空时不显示时间。
  void AddMessage(Role role, std::string text, std::string timestamp = "");
  // 追加文本到最后一条消息（流式输出用）；无消息时忽略。
  void AppendToLast(std::string delta);
  void Clear();

  // 正文折行宽度（显示列），clamp 到 10..200。
  void SetWrapWidth(int cols);
  int WrapWidth() const;

  void SetTheme(const Theme& theme);

  // 当前消息条数（加锁）。
  std::size_t Count() const;

 private:
  ftxui::Element RenderMessage(const Message& m) const;

  // 保护 messages_ / cache_ / 滚动状态（后台流式线程 + UI 线程）。
  mutable std::mutex mutex_;

  std::vector<Message> messages_;

  // 渲染缓存：折行宽度变化时整体重建；新增消息只补建新增项；
  // 流式追加只重建最后一条（dirty_index_ 指向该项）。
  std::vector<ftxui::Element> cache_;
  int cache_wrap_ = -1;
  int dirty_index_ = -1;

  int wrap_width_ = 80;
  Theme theme_;

  // 消息区在屏幕上的矩形（由 Render 的 reflect 写入，UI 线程内使用）。
  ftxui::Box box_;

  // 滚动状态：follow_bottom_ 为真时始终吸附底部；scroll_line_ 为距底部的行数。
  bool follow_bottom_ = true;
  int scroll_line_ = 0;

  // 必须最后声明（析构顺序）。
  ftxui::Component component_;
};

}  // namespace tui
}  // namespace cnn2
