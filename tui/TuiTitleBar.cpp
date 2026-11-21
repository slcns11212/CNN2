// TuiTitleBar.cpp —— 标题栏渲染实现。
#include "TuiTitleBar.h"

#include <utility>

namespace cnn2 {
namespace tui {

TitleBar::TitleBar(std::string title,
                   std::string app_name,
                   std::string status,
                   Theme theme)
    : title_(std::move(title)),
      app_name_(std::move(app_name)),
      status_(std::move(status)),
      theme_(theme) {}

void TitleBar::SetTitle(std::string title) {
  title_ = std::move(title);
}

void TitleBar::SetAppName(std::string app_name) {
  app_name_ = std::move(app_name);
}

void TitleBar::SetStatus(std::string status) {
  status_ = std::move(status);
}

ftxui::Element TitleBar::Render() const {
  using namespace ftxui;

  // 状态颜色编码：运行中 → accent + bold，已中止 → error，其余 → muted。
  Color status_color = theme_.muted;
  bool status_bold = false;
  if (status_ == "运行中…" || status_ == "运行中") {
    status_color = theme_.accent;
    status_bold = true;
  } else if (status_ == "已中止") {
    status_color = theme_.error;
  }
  Element status_text = text(status_) | color(status_color);
  if (status_bold) {
    status_text = status_text | bold;
  }

  // 第一行：应用名（左）与状态（右）。
  Element top = hbox({
      text(app_name_) | color(theme_.muted),
      filler(),
      std::move(status_text),
  });

  // 第二行：标题整宽居中。
  Element title_row = text(title_) | bold | color(theme_.text) | hcenter;

  // 第三行：1 行留白。
  Element gap = text(" ");

  // 第四行：细分隔线。
  Element divider = separatorLight() | color(theme_.divider);

  return vbox({
             top,
             title_row,
             gap,
             divider,
         }) |
         bgcolor(theme_.titlebar);
}

ftxui::Component TitleBar::GetComponent() {
  if (!component_) {
    component_ = ftxui::Renderer([this] { return Render(); });
  }
  return component_;
}

}  // namespace tui
}  // namespace cnn2
