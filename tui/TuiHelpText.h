// TuiHelpText.h —— TUI 指令台帮助文本（单一来源）。
// 同时供 main.cpp 的 `cnn2 tui --help` 与 tui/TuiPage.cpp 的内建 help 使用，
// 保证两处文案完全一致。本头不包含任何其它头文件。
//
// 说明：main.cpp 未加 /utf-8（按系统代码页 936 生成字面量），而 tui/* 编译单元
// 使用 /utf-8，因此同一份源码在不同 TU 里生成的字节数不同。这里刻意用 static
// （内部链接）而非 inline，避免链接器合并 COMDAT 时触发 C4743。
#pragma once

namespace cnn2 {
namespace tui {

static constexpr char kTuiHelpText[] =
    "cnn2 tui —— 终端 TUI 指令台\n"
    "用法: cnn2 tui（或 cnn2 --tui）\n"
    "\n"
    "  在终端全屏界面中直接输入 CNN2 指令并回车执行，输出显示在内容区。\n"
    "  快捷键: Enter 执行 · Tab 循环候选 · 空格进入下一 token · ↑↓ 历史 · Esc 关闭候选 · Ctrl+C 中止/退出\n"
    "\n"
    "内建命令（不进子进程）:\n"
    "  help / ?        显示本帮助\n"
    "  clear / cls     清空内容区\n"
    "  quit / exit     退出 TUI\n"
    "  cd <目录>       切换子进程工作目录\n"
    "  history         显示历史指令\n"
    "  status          显示当前目录与运行状态\n"
    "\n"
    "示例:\n"
    "  version\n"
    "  read model.txt\n"
    "  train data --epochs 10 --gpu\n"
    "  test test/ --model model.txt --fuse soft\n";

}  // namespace tui
}  // namespace cnn2
