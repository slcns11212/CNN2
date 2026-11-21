#pragma once
// 刻意保持最小接口：只暴露画布回调，不引入 conv.h/sf.h/preprocessing.h（GUI 无需这些类型）。
// 注：这些头文件现已全部 inline 化（阶段E E1），即便被包含也不会再触发 LNK2005。
typedef int (*GuiPredictFn)(const unsigned char* canvas, int w, int h, float* conf);
// GUI.cpp 实现：进入消息循环，返回退出码
// fpngPath：可选的 txt 图片路径（32×32 的 '0'/'1' 文本），GUI 启动后加载到画布；
//           为 nullptr 时保持空画布。
int runGUI(GuiPredictFn predict, int canvasSize, const char* fpngPath);
