// TuiConsole.cpp —— Windows 控制台模式/代码页管理与 ANSI 去除。
#include "TuiConsole.h"

#include <windows.h>

namespace cnn2 {
namespace tui {

namespace {

// 句柄是否可用于控制台操作（重定向时可能是 NULL 或 INVALID_HANDLE_VALUE）。
bool IsUsableHandle(HANDLE h) {
  return h != nullptr && h != INVALID_HANDLE_VALUE;
}

}  // namespace

ConsoleState InitializeConsole() {
  ConsoleState state;

  HANDLE in_handle = GetStdHandle(STD_INPUT_HANDLE);
  HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

  // 无论句柄是否有效，始终保存代码页，供 RestoreConsole 还原。
  state.in_cp = GetConsoleCP();
  state.out_cp = GetConsoleOutputCP();

  bool in_valid = IsUsableHandle(in_handle);
  bool out_valid = IsUsableHandle(out_handle);

  DWORD in_mode = 0;
  DWORD out_mode = 0;
  if (in_valid && !GetConsoleMode(in_handle, &in_mode)) {
    in_valid = false;
  }
  if (out_valid && !GetConsoleMode(out_handle, &out_mode)) {
    out_valid = false;
  }

  // 只保存确实读到模式的那一侧（另一侧保持 0，Restore 时不会误用）。
  if (in_valid) {
    state.in_mode = in_mode;
  }
  if (out_valid) {
    state.out_mode = out_mode;
  }

  // UTF-8 代码页可无条件设置（失败静默）。
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);

  // 输出：启用 VT 序列处理，保留原有其它标志。
  if (out_valid) {
    const DWORD new_out_mode = out_mode |
                               ENABLE_VIRTUAL_TERMINAL_PROCESSING |
                               ENABLE_PROCESSED_OUTPUT;
    SetConsoleMode(out_handle, new_out_mode);
  }

  // 输入：关闭快速编辑（避免鼠标选区阻塞），开启扩展标志/鼠标/窗口事件。
  // 注意不设置 ENABLE_VIRTUAL_TERMINAL_INPUT，交给 FTXUI 自行处理。
  if (in_valid) {
    const DWORD new_in_mode =
        (in_mode & ~static_cast<DWORD>(ENABLE_QUICK_EDIT_MODE)) |
        ENABLE_EXTENDED_FLAGS | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT;
    SetConsoleMode(in_handle, new_in_mode);
  }

  // valid 仍表示「输入输出都是控制台」；即使为 false，CP 也已保存并可还原。
  state.valid = in_valid && out_valid;
  return state;
}

void RestoreConsole(const ConsoleState& state) {
  HANDLE in_handle = GetStdHandle(STD_INPUT_HANDLE);
  HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

  // 始终还原代码页（即使当前模式无效/被重定向），失败时静默忽略。
  SetConsoleCP(state.in_cp);
  SetConsoleOutputCP(state.out_cp);

  // 仅对确认为控制台的句柄还原模式；重新探测可避免在重定向句柄上调用。
  DWORD cur_mode = 0;
  if (IsUsableHandle(in_handle) && GetConsoleMode(in_handle, &cur_mode)) {
    SetConsoleMode(in_handle, state.in_mode);
  }
  if (IsUsableHandle(out_handle) && GetConsoleMode(out_handle, &cur_mode)) {
    SetConsoleMode(out_handle, state.out_mode);
  }
}

ConsoleScope::ConsoleScope() : state_(InitializeConsole()) {}

ConsoleScope::~ConsoleScope() { RestoreConsole(state_); }

const ConsoleState& ConsoleScope::State() const { return state_; }

std::string StripAnsi(std::string s) {
  std::string out;
  out.reserve(s.size());

  const std::size_t n = s.size();
  std::size_t i = 0;
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c != 0x1B) {
      // 普通字节（含 UTF-8 多字节序列）原样保留。
      out.push_back(s[i]);
      ++i;
      continue;
    }

    // 遇到 ESC：跳过整个转义序列。
    ++i;
    if (i >= n) {
      break;
    }

    const unsigned char kind = static_cast<unsigned char>(s[i]);
    if (kind == '[') {
      // CSI：ESC [ 参数/中间字节(0x20-0x3F) 终止字节(0x40-0x7E)。
      ++i;
      while (i < n) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        ++i;
        if (b >= 0x40 && b <= 0x7E) {
          break;
        }
      }
    } else if (kind == ']') {
      // OSC：ESC ] ... BEL 或 ESC \（ST）。
      ++i;
      while (i < n) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        if (b == 0x07) {
          ++i;
          break;
        }
        if (b == 0x1B && (i + 1) < n &&
            static_cast<unsigned char>(s[i + 1]) == '\\') {
          i += 2;
          break;
        }
        ++i;
      }
    } else if (kind == '(' || kind == ')' || kind == '*' || kind == '+' ||
               kind == '-' || kind == '.' || kind == '/') {
      // 字符集选择序列是三字节（ESC + 选择符 + 标识符，如 ESC ( B），
      // 需整体丢弃，跳过多出的一个字节。
      ++i;
      if (i < n) {
        ++i;
      }
    } else {
      // 其它两字节序列：ESC + 单个字符。
      ++i;
    }
  }

  return out;
}

}  // namespace tui
}  // namespace cnn2
