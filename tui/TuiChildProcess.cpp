// TuiChildProcess.cpp —— 自身 exe 子进程执行器（匿名管道 + 编码转换 + 取消）。
#include "TuiChildProcess.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cwchar>
#include <string>
#include <utility>
#include <vector>

namespace cnn2 {
namespace tui {

namespace {

// 交付阈值：累积满 8 行或距上次交付超过 50ms（或遇到 '\r'/EOF）时交付一次。
constexpr std::size_t kFlushLines = 8;
constexpr int kFlushIntervalMs = 50;

// 构造子进程环境块：继承父进程环境并注入 CNN2_STDOUT_LINEBUF=1。
// 子进程据此对管道 stdout 启用行缓冲（_IOLBF），否则 CRT 会整块缓冲导致 TUI 输出攒块。
// 返回的缓冲以双 '\0' 结尾，供 CreateProcessW 的 lpEnvironment 使用。
std::vector<wchar_t> BuildChildEnvironment() {
  std::vector<wchar_t> env;
  LPWCH parent = GetEnvironmentStringsW();
  if (parent != nullptr) {
    for (const wchar_t* p = parent; *p != L'\0'; p += std::wcslen(p) + 1) {
      const std::wstring entry(p);
      if (entry.rfind(L"CNN2_STDOUT_LINEBUF=", 0) == 0) {
        continue;  // 去重：父进程若已设置则丢弃，后面统一注入
      }
      env.insert(env.end(), p, p + entry.size() + 1);
    }
    FreeEnvironmentStringsW(parent);
  }
  static const wchar_t kExtra[] = L"CNN2_STDOUT_LINEBUF=1";
  // kExtra 已含结尾 '\0'，一起拷入环境块。
  env.insert(env.end(), kExtra, kExtra + (sizeof(kExtra) / sizeof(wchar_t)));
  // 再加一个 '\0' 形成双 null 结尾。
  env.push_back(L'\0');
  return env;
}

std::wstring Utf8ToWide(const std::string& s) {
  if (s.empty()) {
    return std::wstring();
  }
  const int len = static_cast<int>(s.size());
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), len, nullptr, 0);
  if (n <= 0) {
    return std::wstring();
  }
  std::wstring out(static_cast<std::size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), len, out.data(), n);
  return out;
}

// 子进程 CRT 窄字符输出按系统 ACP（中文系统 936）解释后转成 UTF-8。
std::string AcpToUtf8(const char* data, int len) {
  if (data == nullptr || len <= 0) {
    return std::string();
  }
  const UINT acp = GetACP();
  int wn = MultiByteToWideChar(acp, 0, data, len, nullptr, 0);
  if (wn <= 0) {
    return std::string();
  }
  std::wstring wide(static_cast<std::size_t>(wn), L'\0');
  MultiByteToWideChar(acp, 0, data, len, wide.data(), wn);

  int un = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wn, nullptr, 0, nullptr,
                               nullptr);
  if (un <= 0) {
    return std::string();
  }
  std::string out(static_cast<std::size_t>(un), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), wn, out.data(), un, nullptr,
                      nullptr);
  return out;
}

void CloseIfValid(HANDLE& h) {
  if (h != nullptr && h != INVALID_HANDLE_VALUE) {
    CloseHandle(h);
  }
  h = nullptr;
}

}  // namespace

ChildProcess::ChildProcess() : exe_(SelfExePath()) {}

ChildProcess::~ChildProcess() {
  Cancel();
  CloseIfValid(process_);
  CloseIfValid(thread_);
  CloseIfValid(pipe_read_);
  CloseIfValid(pipe_write_);
}

std::wstring ChildProcess::SelfExePath() {
  DWORD cap = 32768;
  for (int attempt = 0; attempt < 4; ++attempt) {
    std::wstring buf(static_cast<std::size_t>(cap), L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), cap);
    if (n == 0) {
      return std::wstring();
    }
    if (n < cap) {
      buf.resize(n);
      return buf;
    }
    // 缓冲区不足（返回值等于 cap），扩大后重试。
    cap *= 2;
  }
  return std::wstring();
}

// Windows CRT 参数转义规则：含空白/引号时整体加引号；
// 引号前反斜杠加倍；结尾反斜杠加倍。
std::wstring ChildProcess::ArgvQuote(const std::wstring& arg) {
  if (arg.empty()) {
    return L"\"\"";
  }
  if (arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return arg;
  }

  std::wstring out;
  out.push_back(L'"');
  for (auto it = arg.cbegin();; ++it) {
    std::size_t backslashes = 0;
    while (it != arg.cend() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == arg.cend()) {
      out.append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      out.append(backslashes * 2 + 1, L'\\');
      out.push_back(L'"');
    } else {
      out.append(backslashes, L'\\');
      out.push_back(*it);
    }
  }
  out.push_back(L'"');
  return out;
}

bool ChildProcess::Start(const std::vector<std::string>& argv_utf8,
                         const std::wstring& cwd, ChildCallbacks cb) {
  if (busy_.load()) {
    return false;
  }
  // 回收上一轮已结束但尚未 join 的 reader 线程，再覆盖回调。
  if (reader_.joinable()) {
    reader_.join();
  }
  CloseIfValid(process_);
  CloseIfValid(thread_);
  CloseIfValid(pipe_read_);
  CloseIfValid(pipe_write_);

  if (exe_.empty()) {
    return false;
  }
  cb_ = std::move(cb);
  cancelled_.store(false);

  // 命令行：首 token 是自身 exe，后续参数逐个转义。
  std::wstring cmd = ArgvQuote(exe_);
  for (const std::string& arg : argv_utf8) {
    cmd.push_back(L' ');
    cmd += ArgvQuote(Utf8ToWide(arg));
  }
  std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
  cmd_buf.push_back(L'\0');

  // stdin 指向 NUL，避免子进程等待输入而挂起。
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                           OPEN_EXISTING, 0, nullptr);
  if (nul == INVALID_HANDLE_VALUE) {
    nul = nullptr;
  }

  if (!CreatePipe(&pipe_read_, &pipe_write_, &sa, 0)) {
    pipe_read_ = nullptr;
    pipe_write_ = nullptr;
    CloseIfValid(nul);
    return false;
  }
  // 读端不能被子进程继承，否则子进程退出后读端也不会收到 EOF。
  SetHandleInformation(pipe_read_, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  si.hStdInput = nul;
  si.hStdOutput = pipe_write_;
  si.hStdError = pipe_write_;  // stdout/stderr 合并

  PROCESS_INFORMATION pi{};
  // 继承父环境并注入 CNN2_STDOUT_LINEBUF=1；缓冲必须可写，故用 vector::data()。
  std::vector<wchar_t> env_block = BuildChildEnvironment();
  const BOOL ok = CreateProcessW(
      nullptr, cmd_buf.data(), nullptr, nullptr, TRUE,
      CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
      env_block.empty() ? nullptr : env_block.data(),
      cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);

  CloseIfValid(nul);

  if (!ok) {
    CloseIfValid(pipe_read_);
    CloseIfValid(pipe_write_);
    return false;
  }

  process_ = pi.hProcess;
  thread_ = pi.hThread;
  // 父进程必须关掉写端，读端才能看到子进程退出后的 ERROR_BROKEN_PIPE。
  CloseIfValid(pipe_write_);

  busy_.store(true);
  {
    const HANDLE read_handle = pipe_read_;
    reader_ = std::thread([this, read_handle] { ReaderLoop(read_handle); });
  }
  return true;
}

void ChildProcess::ReaderLoop(HANDLE read_handle) {
  std::string pending;  // 尚未按 '\n' 切分的 ACP 原始字节
  std::string out;      // 待交付的 UTF-8 文本
  std::size_t lines = 0;
  auto last_flush = std::chrono::steady_clock::now();

  auto append = [&](const char* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
      if (data[i] == '\n') {
        ++lines;
      }
    }
    out += AcpToUtf8(data, static_cast<int>(len));
  };

  auto flush = [&](bool force) {
    if (out.empty()) {
      return;
    }
    // 注意：这里的间隔检查是"惰性"的——每次 ReadFile 返回时才评估，
    // 不是独立定时器；子进程行缓冲后每行都会写出，ReadFile 随之及时返回。
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - last_flush)
                             .count();
    if (!force && lines < kFlushLines && elapsed < kFlushIntervalMs) {
      return;
    }
    // 去掉 '\r'，避免 TUI 中回车引起光标覆盖。
    std::string chunk;
    chunk.reserve(out.size());
    for (char c : out) {
      if (c != '\r') {
        chunk.push_back(c);
      }
    }
    out.clear();
    lines = 0;
    last_flush = now;
    if (!chunk.empty() && cb_.on_output) {
      cb_.on_output(chunk);
    }
  };

  char buf[4096];
  for (;;) {
    DWORD n = 0;
    if (!ReadFile(read_handle, buf, static_cast<DWORD>(sizeof(buf)), &n,
                  nullptr)) {
      break;  // ERROR_BROKEN_PIPE 及其它错误都视为流结束
    }
    if (n == 0) {
      break;
    }
    pending.append(buf, n);
    // 只转换最后一个 '\n' 或 '\r' 之前的部分，剩余不完整行留到下一轮，
    // 避免多字节字符被读取边界切断。'\r' 也是交付边界（进度条不换行），
    // 交付时仍由 flush 剥离 '\r'，使其表现为多行可见输出。
    const std::size_t nl = pending.rfind('\n');
    const std::size_t cr = pending.rfind('\r');
    std::size_t last = std::string::npos;
    if (nl != std::string::npos && cr != std::string::npos) {
      last = (nl > cr) ? nl : cr;
    } else if (nl != std::string::npos) {
      last = nl;
    } else {
      last = cr;
    }
    if (last != std::string::npos) {
      const std::size_t cut = last + 1;  // 切点必然 >0，pending 一定推进
      const bool ended_with_cr = (pending[cut - 1] == '\r');
      append(pending.data(), cut);
      pending.erase(0, cut);
      // '\r' 结尾的记录（进度行）立即交付，否则等 8 行/50ms 阈值。
      if (ended_with_cr) {
        flush(true);
      }
    }
    flush(false);
  }

  // EOF：转换残留的不完整行并强制交付。
  if (!pending.empty()) {
    append(pending.data(), pending.size());
    pending.clear();
  }
  flush(true);

  if (process_ != nullptr) {
    WaitForSingleObject(process_, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process_, &code);
    const bool cancelled = cancelled_.load();
    if (cb_.on_exit) {
      cb_.on_exit(static_cast<int>(code), cancelled);
    }
  } else if (cb_.on_exit) {
    cb_.on_exit(-1, cancelled_.load());
  }
  busy_.store(false);
}

void ChildProcess::Cancel() {
  cancelled_.store(true);
  if (process_ != nullptr) {
    TerminateProcess(process_, 1);
    WaitForSingleObject(process_, 5000);
  }
  // join 必须在 CloseHandle 之前：reader 线程可能仍在读管道。
  if (reader_.joinable()) {
    reader_.join();
  }
  CloseIfValid(process_);
  CloseIfValid(thread_);
  CloseIfValid(pipe_read_);
  CloseIfValid(pipe_write_);
  busy_.store(false);
}

bool ChildProcess::Busy() const { return busy_.load(); }

}  // namespace tui
}  // namespace cnn2
