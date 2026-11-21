#pragma once

// 统一 FTXUI 入口：以相对路径包含第三方单头实现。
// 注意：本文件只做转发，不引入任何 CNN2 内部头，避免多 TU 重复定义。
#include "../third_party/ftxui/ftxui.hpp"
