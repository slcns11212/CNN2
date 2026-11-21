#pragma once

#include "TuiFtxui.h"

namespace cnn2 {
namespace tui {

// 深色主题调色板（P1 精简版：去掉按钮/滚动条等暂未使用的字段）。
// 默认值统一在 TuiTheme.cpp 的 Theme::Default() 中赋 RGB 常量，
// 头文件里只声明成员，避免大量函数调用形式的静态初始化。
struct Theme {
  ftxui::Color background;
  ftxui::Color panel;
  ftxui::Color titlebar;
  ftxui::Color text;
  ftxui::Color muted;
  ftxui::Color placeholder;
  ftxui::Color accent;
  ftxui::Color system;
  ftxui::Color error;
  ftxui::Color success;
  ftxui::Color border;
  ftxui::Color border_focus;
  ftxui::Color divider;

  static Theme Default();
};

// 装饰器辅助：Decorator 是 std::function<Element(Element)>，
// 不能直接用 | 组合两个 Decorator，因此统一用 lambda 包裹。
ftxui::Decorator DecoratorWithColor(ftxui::Color c);
ftxui::Decorator DecoratorWithBackground(ftxui::Color c);
ftxui::Decorator DecoratorBoldColor(ftxui::Color c);

}  // namespace tui
}  // namespace cnn2
