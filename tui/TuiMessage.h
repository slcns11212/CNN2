#pragma once

#include <string>

namespace cnn2 {
namespace tui {

// P1 仅保留数据结构；ChatView 等渲染逻辑留待后续阶段。
enum class Role {
  User,
  Assistant,
  System,
  Tool,
};

struct Message {
  Role role = Role::System;
  std::string text;
  std::string timestamp;
};

}  // namespace tui
}  // namespace cnn2
