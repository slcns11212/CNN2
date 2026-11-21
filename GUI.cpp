// ============================================================================
// GUI.cpp —— cnn2 的 Win32 手写识别界面（移植自古早 cnn 项目）
// ----------------------------------------------------------------------------
// 纯 Win32 实现：刻意保持最小依赖，不包含 conv.h/sf.h/preprocessing.h（GUI 仅需预测回调）。
// 注：相关头文件现已全部 inline 化（阶段E E1），即便包含也不会再有 LNK2005。
// 仅依赖 user32/gdi32/kernel32；本文件为 UTF-8 with BOM，界面文本用 W 版 API。
// ============================================================================
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include "GUI.h"

#pragma comment(lib, "gdi32.lib")

// 画布参数：32×32，每格 9 像素（288×288），左上角起点 (20,20)
static const int CANVAS = 32;
static const int SCALE = 9;

// ---------------------------------------------------------------- 全局状态
static unsigned char g_canvas[CANVAS * CANVAS] = { 0 };
static bool   g_is_drawing = false;
static bool   g_is_erasing = false;
static int    g_last_cx = -1;
static int    g_last_cy = -1;
static int    g_brush = 1;
static HWND   g_hWnd = NULL;
static HWND   g_static_res = NULL;
static HBRUSH g_bgBrush = NULL;    // 画布黑底刷：只创建一次，退出时释放
static GuiPredictFn g_predict = NULL;

static void clear_canvas() {
    memset(g_canvas, 0, sizeof(g_canvas));
    InvalidateRect(g_hWnd, NULL, TRUE);
    if (g_static_res != NULL)
        SetWindowTextW(g_static_res, L"识别结果：未识别");
    std::cout << "画布已清空" << std::endl;
}

// 从 txt 图片文件加载到画布：每行取前 32 个字符，'1' -> 255、'0' -> 0。
// 最多 32 行 × 32 列；空行/短行跳过或补 0；文件打不开仅打印错误，不影响窗口。
static void load_fpng_to_canvas(const char* path) {
    if (path == NULL || path[0] == '\0') return;

    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "GUI: 无法打开 --fpng 文件: " << path << std::endl;
        return;
    }

    memset(g_canvas, 0, sizeof(g_canvas));

    std::string line;
    int y = 0;
    while (y < CANVAS && std::getline(fin, line)) {
        // 兼容 CRLF：去掉行尾的 '\r'
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) { ++y; continue; } // 空行按补 0 处理

        int n = (int)line.size();
        for (int x = 0; x < CANVAS && x < n; ++x) {
            if (line[x] == '1') g_canvas[y * CANVAS + x] = 255;
            // '0' 或其他字符保持 0
        }
        ++y;
    }

    std::cout << "GUI: 已从 " << path << " 加载画布（"
        << y << " 行）" << std::endl;
    InvalidateRect(g_hWnd, NULL, TRUE);
}

static void draw_brush(int x, int y, unsigned char val) {
    int hw = g_brush / 2;
    int x1 = max(0, x - hw);
    int x2 = min(CANVAS - 1, x + (g_brush - 1 - hw));
    int y1 = max(0, y - hw);
    int y2 = min(CANVAS - 1, y + (g_brush - 1 - hw));
    for (int cy = y1; cy <= y2; cy++)
        for (int cx = x1; cx <= x2; cx++)
            g_canvas[cy * CANVAS + cx] = val;
}

static void draw_line(int x1, int y1, int x2, int y2, unsigned char val) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    while (1) {
        if (x1 >= 0 && x1 < CANVAS && y1 >= 0 && y1 < CANVAS)
            draw_brush(x1, y1, val);
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

static void recognize() {
    std::cout << "\n===== 开始识别 =====" << std::endl;
    if (g_predict == NULL) {
        if (g_static_res != NULL)
            SetWindowTextW(g_static_res, L"识别结果：未加载模型");
        return;
    }
    float conf = 0.0f;
    int res = g_predict(g_canvas, CANVAS, CANVAS, &conf);
    std::cout << "识别结果：数字 " << res << "  置信度："
        << conf * 100.0f << "%" << std::endl;

    if (g_static_res != NULL) {
        wchar_t res_text[100];
        swprintf_s(res_text, L"识别结果：数字 %d（置信度 %.2f%%）",
            res, conf * 100.0f);
        SetWindowTextW(g_static_res, res_text);
    }
}

static void on_paint(HWND hWnd) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hWnd, &ps);

    // 画布黑底：使用只创建一次的背景刷
    RECT bg = { 20, 20, 20 + CANVAS * SCALE, 20 + CANVAS * SCALE };
    FillRect(hdc, &bg, g_bgBrush);

    // 白色前景刷：每帧创建、用完即删
    HBRUSH white = CreateSolidBrush(RGB(255, 255, 255));
    for (int y = 0; y < CANVAS; y++) {
        for (int x = 0; x < CANVAS; x++) {
            if (g_canvas[y * CANVAS + x]) {
                RECT r = {
                    20 + x * SCALE,
                    20 + y * SCALE,
                    20 + (x + 1) * SCALE,
                    20 + (y + 1) * SCALE
                };
                FillRect(hdc, &r, white);
            }
        }
    }
    DeleteObject(white);

    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(255, 255, 255));
    wchar_t buf[100];
    swprintf_s(buf, L"画笔：%d | 左键画 | 右键擦 | 滚轮调笔刷", g_brush);
    TextOutW(hdc, 20, 20 + CANVAS * SCALE + 12, buf, (int)wcslen(buf));

    EndPaint(hWnd, &ps);
}

static bool screen2canvas(int x, int y, int& cx, int& cy) {
    cx = (x - 20) / SCALE;
    cy = (y - 20) / SCALE;
    return cx >= 0 && cx < CANVAS && cy >= 0 && cy < CANVAS;
}

static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    int cx = 0, cy = 0;
    switch (msg) {
    case WM_PAINT:
        on_paint(hWnd);
        return 0;
    case WM_LBUTTONDOWN:
        g_is_drawing = true;
        g_is_erasing = false;
        if (screen2canvas(LOWORD(lParam), HIWORD(lParam), cx, cy)) {
            g_last_cx = cx; g_last_cy = cy;
            draw_brush(cx, cy, 255);
            InvalidateRect(hWnd, NULL, TRUE);
        }
        return 0;
    case WM_RBUTTONDOWN:
        g_is_erasing = true;
        g_is_drawing = false;
        if (screen2canvas(LOWORD(lParam), HIWORD(lParam), cx, cy)) {
            g_last_cx = cx; g_last_cy = cy;
            draw_brush(cx, cy, 0);
            InvalidateRect(hWnd, NULL, TRUE);
        }
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        g_is_drawing = g_is_erasing = false;
        g_last_cx = g_last_cy = -1;
        return 0;
    case WM_MOUSEMOVE:
        if (!g_is_drawing && !g_is_erasing) return 0;
        if (screen2canvas(LOWORD(lParam), HIWORD(lParam), cx, cy) && g_last_cx != -1) {
            draw_line(g_last_cx, g_last_cy, cx, cy, g_is_drawing ? 255 : 0);
            g_last_cx = cx; g_last_cy = cy;
            InvalidateRect(hWnd, NULL, TRUE);
        }
        return 0;
    case WM_MOUSEWHEEL:
        g_brush = (short)GET_WHEEL_DELTA_WPARAM(wParam) > 0
            ? min(g_brush + 1, 8)
            : max(g_brush - 1, 1);
        InvalidateRect(hWnd, NULL, TRUE);
        return 0;
    case WM_COMMAND:
        if (LOWORD(wParam) == 1) recognize();
        if (LOWORD(wParam) == 2) clear_canvas();
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

static void create_controls(HWND hWnd) {
    CreateWindowW(L"BUTTON", L"识别画布",
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        370, 100, 120, 35, hWnd, (HMENU)1, NULL, NULL);
    CreateWindowW(L"BUTTON", L"清空画布",
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        370, 150, 120, 35, hWnd, (HMENU)2, NULL, NULL);
    g_static_res = CreateWindowW(L"STATIC", L"识别结果：未识别",
        WS_VISIBLE | WS_CHILD | SS_LEFT | SS_CENTERIMAGE,
        370, 50, 240, 30, hWnd, NULL, NULL, NULL);
}

int runGUI(GuiPredictFn predict, int canvasSize, const char* fpngPath) {
    if (predict == NULL) {
        std::cerr << "GUI: 预测回调为空\n";
        return 1;
    }
    if (canvasSize != CANVAS) {
        std::cerr << "GUI: 画布尺寸必须为 " << CANVAS
            << "，收到 " << canvasSize << "\n";
        return 1;
    }
    g_predict = predict;

    HINSTANCE hInst = GetModuleHandleW(NULL);
    const wchar_t* cls = L"cnn2gui";

    HBRUSH classBrush = CreateSolidBrush(RGB(32, 32, 32));
    WNDCLASSEXW wc = { 0 };
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.hbrBackground = classBrush;
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hIcon = LoadIconW(NULL, IDI_APPLICATION);
    wc.hIconSm = LoadIconW(NULL, IDI_APPLICATION);

    if (!RegisterClassExW(&wc)) {
        std::cerr << "窗口类注册失败！错误码：" << GetLastError() << std::endl;
        DeleteObject(classBrush);
        return -1;
    }

    g_hWnd = CreateWindowExW(0, cls, L"atcm",
        WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX,
        100, 100, 650, 400, NULL, NULL, hInst, NULL);
    if (g_hWnd == NULL) {
        std::cerr << "主窗口创建失败！错误码：" << GetLastError() << std::endl;
        DeleteObject(classBrush);
        return -1;
    }

    g_bgBrush = CreateSolidBrush(RGB(0, 0, 0)); // 只创建一次

    create_controls(g_hWnd);
    ShowWindow(g_hWnd, SW_SHOW);
    UpdateWindow(g_hWnd);

    // 窗口就绪后、进入消息循环前，把 --fpng 指定的 txt 图片加载到画布
    load_fpng_to_canvas(fpngPath);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (g_bgBrush) { DeleteObject(g_bgBrush); g_bgBrush = NULL; }
    DeleteObject(classBrush);
    return 0;
}
