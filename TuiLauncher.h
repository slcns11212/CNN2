#pragma once

// TUI 启动入口（纯 C 接口，便于在 main.cpp 中调用而不引入 C++ 头依赖）。
#ifdef __cplusplus
extern "C" {
#endif

// 进入 TUI 全屏界面，返回退出码。
// knownKeys 为参数键的 C 视图（不含 -- 前缀），会注入补全器以保持与 CLI 同源。
int runTuiMode(int argc, char** argv, const char* const* knownKeys,
               int knownKeyCount);

#ifdef __cplusplus
}
#endif
