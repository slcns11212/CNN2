// TuiPage.cpp —— 整页布局：标题栏 + 消息流 + 分隔线 + 输入栏，
// 以及内建指令分发与 CNN2 子进程执行。
#include "TuiPage.h"

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "TuiHelpText.h"  // 与 main.cpp 的 tui --help 共用同一份帮助文案

namespace cnn2 {
namespace tui {

namespace {

std::string TrimAscii(const std::string& s) {
  std::size_t begin = 0;
  std::size_t end = s.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
    --end;
  }
  return s.substr(begin, end - begin);
}

std::string WideToUtf8(const std::wstring& w) {
  if (w.empty()) {
    return std::string();
  }
  const int len = static_cast<int>(w.size());
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), len, nullptr, 0, nullptr,
                              nullptr);
  if (n <= 0) {
    return std::string();
  }
  std::string out(static_cast<std::size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), len, out.data(), n, nullptr,
                      nullptr);
  return out;
}

std::wstring CurrentDirectory() {
  const DWORD need = GetCurrentDirectoryW(0, nullptr);
  if (need == 0) {
    return L".";
  }
  std::wstring buf(static_cast<std::size_t>(need), L'\0');
  const DWORD got = GetCurrentDirectoryW(need, buf.data());
  if (got == 0) {
    return L".";
  }
  buf.resize(got);
  return buf;
}

// 简单分词：空白分隔，支持双引号包裹（引号内的 \" 视为字面引号，
// 反斜杠路径在引号外原样保留）。
std::vector<std::string> ParseCommandLineUtf8(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quote = false;
  bool has_token = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quote) {
      if (c == '\\' && i + 1 < line.size() && line[i + 1] == '"') {
        cur.push_back('"');
        ++i;
      } else if (c == '"') {
        in_quote = false;
      } else {
        cur.push_back(c);
      }
      continue;
    }
    if (c == '"') {
      in_quote = true;
      has_token = true;
    } else if (c == ' ' || c == '\t') {
      if (has_token || !cur.empty()) {
        out.push_back(cur);
        cur.clear();
        has_token = false;
      }
    } else {
      cur.push_back(c);
      has_token = true;
    }
  }
  if (has_token || !cur.empty()) {
    out.push_back(cur);
  }
  return out;
}

// 逐字段构造 InputBarOptions，不能先构造临时 InputBar 再移动
// （InputBar 内部的 ftxui::Input 持有 draft_ 裸指针）。
InputBarOptions MakeInputOptions(const TuiPageOptions& opt) {
  InputBarOptions ib;
  ib.placeholder = opt.placeholder;
  ib.hint = "Enter 执行 · Tab 补全 · ↑↓ 历史 · Ctrl+C 中止/退出";
  ib.show_send_button = false;
  ib.theme = opt.theme;
  return ib;
}

}  // namespace

TuiPage::TuiPage(TuiPageOptions options)
    : opt_(std::move(options)),
      title_(opt_.title, opt_.app_name, opt_.status, opt_.theme),
      input_(MakeInputOptions(opt_)),
      cwd_(CurrentDirectory()),
      status_(opt_.status) {
  // ChatView 默认主题与页面主题保持一致（构建组件前设置，避免旧主题缓存）。
  view_.SetTheme(opt_.theme);

  // 组装组件树：ChatView 不参与 Tab 焦点切换，焦点由输入栏独占。
  auto body = ftxui::Container::Vertical({
      view_.GetComponent(),
      input_.GetComponent(),
  });
  root_ = ftxui::Renderer(body, [this] { return Render(); });

  // 输入栏提交统一交给 HandleSubmit：内建指令优先，其余走子进程。
  input_.SetOnSubmit([this](std::string text) {
    HandleSubmit(std::move(text));
  });

  // 即使未注入 knownKeys，也启用命令词 / 内建键表补全。
  input_.SetCompleter(&completer_);

  // 容器默认聚焦第 0 个子项（不可聚焦的 ChatView），这里显式把焦点交给输入框。
  input_.Focus();
}

ftxui::Component TuiPage::GetComponent() { return root_; }

void TuiPage::SetOnSubmit(std::function<void(std::string)> cb) {
  on_submit_ = std::move(cb);
}

void TuiPage::AddMessage(Role role, std::string text) {
  view_.AddMessage(role, std::move(text));
}

void TuiPage::AppendToLast(std::string delta) {
  view_.AppendToLast(std::move(delta));
}

void TuiPage::ClearMessages() { view_.Clear(); }

// 折行宽度需扣除消息正文缩进（4 列）与竖直滚动条（1 列）。
void TuiPage::SetWrapWidth(int cols) {
  const int usable = std::max(10, cols - 5);
  view_.SetWrapWidth(usable);
}

void TuiPage::AddLine(std::string line) {
  view_.AddMessage(Role::System, std::move(line));
}

void TuiPage::RequestRedraw() {
  if (redraw_) {
    redraw_();
  }
}

void TuiPage::SetRedrawRequest(std::function<void()> cb) {
  redraw_ = std::move(cb);
}

void TuiPage::FocusInput() { input_.Focus(); }

void TuiPage::SetExitRequest(std::function<void()> cb) {
  exit_request_ = std::move(cb);
}

void TuiPage::RequestExit() {
  if (exit_request_) {
    exit_request_();
  }
}

bool TuiPage::IsChildBusy() const { return child_.Busy(); }

void TuiPage::SetStatus(std::string status) {
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = std::move(status);
  }
  RequestRedraw();
}

void TuiPage::SetKnownKeys(const char* const* keys, int count) {
  // 补全器与本地存储共用同一份键表；无论是否为空都绑定补全器，
  // 以保证命令词补全可用。
  completer_.SetKnownKeys(keys, count);
  input_.SetCompleter(&completer_);

  known_keys_.clear();
  if (keys == nullptr || count <= 0) {
    return;
  }
  known_keys_.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (keys[i] != nullptr) {
      known_keys_.emplace_back(keys[i]);
    }
  }
}

void TuiPage::HandleSubmit(std::string line) {
  const std::string text = TrimAscii(line);
  if (text.empty()) {
    return;
  }

  // 先记历史（含内建指令），再分发执行。
  completer_.AddHistory(text);

  AddMessage(Role::User, text);

  // 保留 P2 的 SetOnSubmit 兼容：外部观察者仍能收到提交内容（可选）。
  if (on_submit_) {
    on_submit_(text);
  }

  // 内建指令命中则不进入子进程。
  if (DispatchBuiltin(text)) {
    RequestRedraw();
    return;
  }
  RunChild(text);
}

bool TuiPage::DispatchBuiltin(const std::string& line) {
  const std::size_t sp = line.find_first_of(" \t");
  const std::string cmd =
      line.substr(0, sp == std::string::npos ? line.size() : sp);
  const std::string rest =
      (sp == std::string::npos) ? std::string() : TrimAscii(line.substr(sp + 1));

  // 注意：内建指令名大小写敏感（仅小写）。
  if (cmd == "help" || cmd == "?") {
    AddMessage(Role::System, kTuiHelpText);
    return true;
  }
  if (cmd == "clear" || cmd == "cls") {
    view_.Clear();
    AddMessage(Role::System, "内容区已清空");
    return true;
  }
  if (cmd == "quit" || cmd == "exit") {
    if (child_.Busy()) {
      AddMessage(Role::System, "子进程仍在运行，请等待其结束或中止后再退出");
      return true;
    }
    if (exit_request_) {
      exit_request_();
    } else {
      AddMessage(Role::System, "当前未设置退出回调，无法退出 TUI");
    }
    return true;
  }
  if (cmd == "cd") {
    if (rest.empty()) {
      AddMessage(Role::System, "当前目录: " + WideToUtf8(cwd_));
      return true;
    }
    std::filesystem::path target = std::filesystem::u8path(rest);
    if (target.is_relative()) {
      target = std::filesystem::path(cwd_) / target;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(target, ec) || ec) {
      AddMessage(Role::System, "目录不存在: " + rest);
      return true;
    }
    std::filesystem::path canon = std::filesystem::weakly_canonical(target, ec);
    if (ec) {
      canon = target;
    }
    cwd_ = canon.wstring();
    AddMessage(Role::System, "当前目录: " + WideToUtf8(cwd_));
    return true;
  }
  if (cmd == "history") {
    const std::vector<std::string>& hist = completer_.History();
    if (hist.empty()) {
      AddMessage(Role::System, "（无历史）");
      return true;
    }
    std::string out;
    for (std::size_t i = 0; i < hist.size(); ++i) {
      if (!out.empty()) {
        out += '\n';
      }
      out += std::to_string(i + 1);
      out += "  ";
      out += hist[i];
    }
    AddMessage(Role::System, out);
    return true;
  }
  if (cmd == "status") {
    std::string text = "目录: " + WideToUtf8(cwd_);
    text += "\n子进程: ";
    text += child_.Busy() ? "运行中" : "空闲";
    text += "\n消息数: " + std::to_string(view_.Count());
    AddMessage(Role::System, text);
    return true;
  }
  return false;
}

void TuiPage::EnsureOutputMessage() {
  // exchange 保证「检查 + 设置」原子化：并发的首块输出只会开一条消息。
  if (!output_open_.exchange(true)) {
    AddMessage(Role::Tool, "");
  }
}

void TuiPage::RunChild(const std::string& line) {
  if (child_.Busy()) {
    AddMessage(Role::System, "上一条指令仍在运行");
    RequestRedraw();
    return;
  }

  std::vector<std::string> argv = ParseCommandLineUtf8(line);
  if (argv.empty()) {
    return;
  }
  if (argv[0] == "tui" || argv[0] == "--tui") {
    AddMessage(Role::System, "已在 TUI 中，不能再次启动 TUI。");
    RequestRedraw();
    return;
  }
  if (argv[0] == "gui") {
    AddMessage(Role::System, "提示: gui 会打开独立窗口，TUI 将在其运行期间保持等待");
  }

  output_open_.store(false);
  SetStatus("运行中…");

  ChildCallbacks cb;
  cb.on_output = [this](const std::string& chunk) {
    EnsureOutputMessage();
    AppendToLast(chunk);
    RequestRedraw();
  };
  cb.on_exit = [this](int code, bool cancelled) {
    // on_exit 在 reader 线程触发：AddMessage/SetStatus 内部各自加锁，
    // RequestRedraw 走 ScreenInteractive::PostEvent，均可跨线程调用。
    output_open_.store(false);
    AddMessage(Role::System,
               cancelled ? "[已中止]"
                         : ("[退出码 " + std::to_string(code) + "]"));
    SetStatus(cancelled ? "已中止" : "就绪");
    RequestRedraw();
  };

  if (!child_.Start(argv, cwd_, std::move(cb))) {
    output_open_.store(false);
    SetStatus("就绪");
    AddMessage(Role::System, "启动子进程失败");
  }
  RequestRedraw();
}

void TuiPage::CancelChild() {
  if (!child_.Busy()) {
    return;
  }
  // Cancel 内部会 join reader 线程，on_exit 已在 join 前触发（会写入中止消息）。
  child_.Cancel();
  // on_exit 已把状态置为“已中止”，这里再显式设置一次，保证 UI 线程侧
  // 观察到确定状态（Cancel 可能发生在 on_exit 之后才返回）。
  SetStatus("已中止");
  RequestRedraw();
}

ftxui::Element TuiPage::Render() {
  using namespace ftxui;

  Elements parts;
  if (opt_.show_title_bar) {
    // 状态由 TuiPage 加锁维护，UI 线程在这里同步给标题栏。
    std::string status;
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      status = status_;
    }
    title_.SetStatus(std::move(status));
    parts.push_back(title_.Render());
  }

  // 消息流区域占满剩余高度。
  parts.push_back(view_.Render() | flex);

  parts.push_back(separatorLight() | color(opt_.theme.divider));
  parts.push_back(input_.Render());

  return vbox(std::move(parts)) | bgcolor(opt_.theme.background) |
         color(opt_.theme.text);
}

int RunTuiPage(TuiPage& page) {
  // 用 shared_ptr 持有 screen，配合 weak_ptr 回调：即使后台线程在
  // ScreenInteractive 析构后仍持有旧回调，weak.lock() 也会失败，不会悬垂。
  auto screen = std::make_shared<ftxui::ScreenInteractive>(
      ftxui::ScreenInteractive::Fullscreen());
  std::weak_ptr<ftxui::ScreenInteractive> weak = screen;
  page.SetRedrawRequest([weak] {
    if (auto s = weak.lock()) {
      s->PostEvent(ftxui::Event::Custom);
    }
  });
  // quit/exit 内建指令通过 ScreenInteractive::Exit 结束主循环。
  page.SetExitRequest([weak] {
    if (auto s = weak.lock()) {
      s->Exit();
    }
  });

  // 先构建组件，再设置焦点，最后进入循环。
  auto component = page.GetComponent();
  page.FocusInput();

  // 折行宽度与终端联动：读取当前控制台窗口列数，扣掉正文缩进与滚动条后
  // 交给消息流，避免每行末尾被裁掉 4 列（中文跨边界错位）。
  int cols = 100;
  if (HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
      out != nullptr && out != INVALID_HANDLE_VALUE) {
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (GetConsoleScreenBufferInfo(out, &info) != 0) {
      cols = static_cast<int>(info.srWindow.Right - info.srWindow.Left) + 1;
    }
  }
  page.SetWrapWidth(cols);

  // Ctrl+C 双语义：
  //   1) 有子进程在跑 → 中止子进程（消费事件，不退出）；
  //   2) 空闲 → 主动请求退出主循环（而非返回 false 让 FTXUI raise(SIGINT)），
  //      这样 ScreenInteractive 正常析构，ConsoleScope 恢复代码页/输入模式。
  // CatchEvent 的 handler 返回 false 时 ComponentBase::OnEvent 会把事件转发
  // 给被包裹的 child，因此焦点/按键转发不受影响。
  // 注意：App 默认 force_handle_ctrl_c_ = true（见 third_party/ftxui 中
  // App::Internal::HandleTask：`if (arg == Event::CtrlC && (!handled ||
  // force_handle_ctrl_c_)) RecordSignal(SIGINT);`），即组件即使消费了事件，
  // FTXUI 仍会退出。故必须先关闭强制处理，让“未消费”才退出。
  screen->ForceHandleCtrlC(false);
  auto guarded = ftxui::CatchEvent(
      component, [&page](ftxui::Event e) {
        if (e != ftxui::Event::CtrlC) {
          return false;
        }
        if (page.IsChildBusy()) {
          page.CancelChild();
          return true;
        }
        // 空闲：请求退出，走 screen->Exit() 的正常退出路径。
        page.RequestExit();
        return true;
      });
  screen->Loop(guarded);

  // 退出循环后解除回调，避免后续无谓的 PostEvent；再中止残留子进程。
  page.SetRedrawRequest(nullptr);
  page.SetExitRequest(nullptr);
  page.CancelChild();
  return 0;
}

}  // namespace tui
}  // namespace cnn2
