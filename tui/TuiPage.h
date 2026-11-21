#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "TuiChatView.h"
#include "TuiChildProcess.h"
#include "TuiCommandCompleter.h"
#include "TuiInputBar.h"
#include "TuiMessage.h"
#include "TuiTheme.h"
#include "TuiTitleBar.h"

namespace cnn2 {
namespace tui {

struct TuiPageOptions {
  std::string title = "CNN2 TUI";
  std::string app_name = "CNN2";
  std::string status = "就绪";
  std::string placeholder = "输入指令…";
  bool show_title_bar = true;
  Theme theme = Theme::Default();
};

// 整页布局：标题栏 + 消息流（ChatView） + 分隔线 + 输入栏。
class TuiPage {
 public:
  explicit TuiPage(TuiPageOptions options = {});

  // 内部 Component 回调捕获了 this，拷贝/移动会导致悬垂指针。
  TuiPage(const TuiPage&) = delete;
  TuiPage& operator=(const TuiPage&) = delete;
  TuiPage(TuiPage&&) = delete;
  TuiPage& operator=(TuiPage&&) = delete;

  ftxui::Component GetComponent();

  void SetOnSubmit(std::function<void(std::string)> cb);

  // 转发到内部 ChatView。
  void AddMessage(Role role, std::string text);
  void AppendToLast(std::string delta);
  void ClearMessages();
  void SetWrapWidth(int cols);

  // 兼容 P1：等价于 AddMessage(Role::System, line)。
  void AddLine(std::string line);

  void RequestRedraw();
  void SetRedrawRequest(std::function<void()> cb);
  void FocusInput();

  // 输入栏提交入口：内建指令优先，否则启动子进程执行 CNN2 命令。
  void HandleSubmit(std::string line);

  // quit/exit 的退出动作（由 RunTuiPage 注入 ScreenInteractive::Exit）。
  void SetExitRequest(std::function<void()> cb);

  // 主动请求退出主循环（空闲 Ctrl+C 使用）：内部转调 exit_request_，
  // 使程序走正常析构路径恢复终端（代码页/输入模式）。
  void RequestExit();

  // 子进程状态与取消（Ctrl+C 等场景由外部调用）。
  bool IsChildBusy() const;
  void CancelChild();

  // 标题栏状态（线程安全，可在 reader 线程调用）。
  void SetStatus(std::string status);

  // P5 补全用：保存 CLI 已知参数键（P3 仅存储占位）。
  void SetKnownKeys(const char* const* keys, int count);

 private:
  ftxui::Element Render();
  bool DispatchBuiltin(const std::string& line);
  void RunChild(const std::string& line);
  void EnsureOutputMessage();

  TuiPageOptions opt_;
  std::function<void(std::string)> on_submit_;
  TitleBar title_;
  ChatView view_;
  // 必须声明在 input_ 之前：input_ 持有 &completer_，逆序析构时
  // input_ 先销毁，completer_ 仍在生命周期内。
  CommandCompleter completer_;
  InputBar input_;

  std::wstring cwd_;

  mutable std::mutex status_mutex_;
  std::string status_;
  // 由 UI 线程（EnsureOutputMessage）与 reader 线程（on_exit）并发访问。
  std::atomic<bool> output_open_{false};
  std::function<void()> exit_request_;
  std::vector<std::string> known_keys_;

  // 由 RunTuiPage 注入（screen.PostEvent），供后台线程请求重绘。
  std::function<void()> redraw_;

  // child_ 必须在 title_/view_/input_ 之后声明：逆序析构时子进程先被
  // Cancel/join，回调访问到的视图与标题栏仍然有效。同时必须位于
  // redraw_/status_mutex_ 之前析构——故紧邻 root_ 之前声明。
  ChildProcess child_;

  // 必须最后声明（析构顺序）。
  ftxui::Component root_;
};

// 启动全屏 TUI 主循环，返回进程退出码。
int RunTuiPage(TuiPage& page);

}  // namespace tui
}  // namespace cnn2
