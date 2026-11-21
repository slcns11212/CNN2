#pragma once

// 子进程执行器：以自身 exe 为目标，stdout/stderr 合并到匿名管道，
// 按完整行缓冲后做 ACP → UTF-8 转换再交付。
// 注意：本头文件只包含 Windows API 与标准库，不包含任何 CNN2 内部头。

#include <windows.h>

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace cnn2 {
namespace tui {

struct ChildCallbacks {
  // 已转 UTF-8、已按完整行合并的输出块（在 reader 线程调用）。
  std::function<void(const std::string& chunk)> on_output;
  // 进程结束（在 reader 线程调用）。
  std::function<void(int exit_code, bool cancelled)> on_exit;
};

class ChildProcess {
 public:
  ChildProcess();
  ~ChildProcess();  // 内部 Cancel + join + CloseHandle

  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  // argv_utf8 不含 exe 路径，首元素为子命令（如 "version"）。
  bool Start(const std::vector<std::string>& argv_utf8,
             const std::wstring& cwd, ChildCallbacks cb);
  void Cancel();
  bool Busy() const;

 private:
  static std::wstring SelfExePath();
  static std::wstring ArgvQuote(const std::wstring& arg);
  void ReaderLoop(HANDLE read_handle);

  std::wstring exe_;
  HANDLE process_ = nullptr;
  HANDLE thread_ = nullptr;
  HANDLE pipe_read_ = nullptr;
  HANDLE pipe_write_ = nullptr;
  std::thread reader_;
  std::atomic<bool> busy_{false};
  std::atomic<bool> cancelled_{false};
  ChildCallbacks cb_;
};

}  // namespace tui
}  // namespace cnn2
