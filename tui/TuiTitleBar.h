#pragma once

#include <string>

#include "TuiTheme.h"

namespace cnn2 {
namespace tui {

// 标题栏：应用名（左）+ 状态（右）、居中标题、留白与分隔线。
class TitleBar {
 public:
  explicit TitleBar(std::string title = "CNN2 TUI",
                    std::string app_name = "CNN2",
                    std::string status = "",
                    Theme theme = Theme::Default());

  // 内部 ftxui::Component 回调捕获了 this，拷贝/移动会导致悬垂指针。
  TitleBar(const TitleBar&) = delete;
  TitleBar& operator=(const TitleBar&) = delete;
  TitleBar(TitleBar&&) = delete;
  TitleBar& operator=(TitleBar&&) = delete;

  void SetTitle(std::string title);
  void SetAppName(std::string app_name);
  void SetStatus(std::string status);

  ftxui::Element Render() const;
  ftxui::Component GetComponent();

 private:
  std::string title_;
  std::string app_name_;
  std::string status_;
  Theme theme_;

  // 必须最后声明（析构顺序）。
  ftxui::Component component_;
};

}  // namespace tui
}  // namespace cnn2
