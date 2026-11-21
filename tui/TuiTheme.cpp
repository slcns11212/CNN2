// TuiTheme.cpp —— 主题常量与装饰器实现。
#include "TuiTheme.h"

namespace cnn2 {
namespace tui {

Theme Theme::Default() {
  Theme t;
  t.background = ftxui::Color::RGB(14, 14, 16);
  t.panel = ftxui::Color::RGB(23, 23, 26);
  t.titlebar = ftxui::Color::RGB(18, 18, 21);
  t.text = ftxui::Color::RGB(228, 228, 231);
  t.muted = ftxui::Color::RGB(161, 161, 170);
  t.placeholder = ftxui::Color::RGB(110, 110, 120);
  t.accent = ftxui::Color::RGB(232, 168, 124);
  t.system = ftxui::Color::RGB(184, 168, 138);
  t.error = ftxui::Color::RGB(217, 115, 107);
  t.success = ftxui::Color::RGB(134, 181, 138);
  t.border = ftxui::Color::RGB(42, 42, 47);
  t.border_focus = ftxui::Color::RGB(154, 126, 99);
  t.divider = ftxui::Color::RGB(31, 31, 35);
  return t;
}

ftxui::Decorator DecoratorWithColor(ftxui::Color c) {
  return [c](ftxui::Element e) { return e | ftxui::color(c); };
}

ftxui::Decorator DecoratorWithBackground(ftxui::Color c) {
  return [c](ftxui::Element e) { return e | ftxui::bgcolor(c); };
}

ftxui::Decorator DecoratorBoldColor(ftxui::Color c) {
  return [c](ftxui::Element e) { return e | ftxui::color(c) | ftxui::bold; };
}

}  // namespace tui
}  // namespace cnn2
