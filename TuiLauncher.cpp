// TuiLauncher.cpp —— TUI 模式启动封装。
// 注意：不要包含任何 CNN2 内部头（conv.h/sf.h/preprocessing.h 等），
// 这些头含非 inline 全局定义，只允许 main.cpp 一个 TU 包含，否则 LNK2005。
#include "TuiLauncher.h"

#include <utility>

#include "tui/TuiConsole.h"
#include "tui/TuiPage.h"

extern "C" int runTuiMode(int argc, char** argv, const char* const* knownKeys,
                          int knownKeyCount) {
  // P1 暂不使用命令行参数（仅用于进入 TUI 模式本身）。
  (void)argc;
  (void)argv;

  // 先声明 ConsoleScope：page 后声明，先析构，保证退出时先恢复页面再恢复终端。
  cnn2::tui::ConsoleScope console;

  cnn2::tui::TuiPageOptions opt;
  opt.title = "CNN2 TUI";
  opt.app_name = "CNN2";
  opt.status = "就绪";
  opt.placeholder = "输入指令…";

  cnn2::tui::TuiPage page(std::move(opt));
  page.AddMessage(cnn2::tui::Role::System, "欢迎使用 CNN2 TUI。输入指令后回车。");

  // 把 main 的 knownArgKeys() 键表注入补全器，保证 TUI 补全与 CLI 参数同源。
  if (knownKeys != nullptr && knownKeyCount > 0) {
    page.SetKnownKeys(knownKeys, knownKeyCount);
  }

  // P1 回显逻辑由 TuiPage 内部处理（未设置 on_submit 时自动 "已收到: ..."）。
  return cnn2::tui::RunTuiPage(page);
}
