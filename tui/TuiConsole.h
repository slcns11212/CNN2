#pragma once

#include <string>

namespace cnn2 {
namespace tui {

// 控制台模式 / 代码页快照。
// 不在头文件中包含 <windows.h>：DWORD 用 unsigned long、UINT 用 unsigned int
// （Windows ABI 与目标平台一致）。
struct ConsoleState {
  bool valid = false;
  unsigned long in_mode = 0;
  unsigned long out_mode = 0;
  unsigned int in_cp = 0;
  unsigned int out_cp = 0;
};

ConsoleState InitializeConsole();
void RestoreConsole(const ConsoleState& state);

// RAII：构造时保存并初始化控制台（UTF-8 代码页、输出 VT、输入鼠标/窗口事件），
// 析构时恢复。
class ConsoleScope {
 public:
  ConsoleScope();
  ~ConsoleScope();
  ConsoleScope(const ConsoleScope&) = delete;
  ConsoleScope& operator=(const ConsoleScope&) = delete;

  const ConsoleState& State() const;

 private:
  ConsoleState state_;
};

// 去掉 ANSI 转义序列（ESC[...m 等）。
std::string StripAnsi(std::string s);

}  // namespace tui
}  // namespace cnn2
