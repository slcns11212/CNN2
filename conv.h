#pragma once
#include <vector>

// ================= ConvMap =================
class ConvMap {
public:
    int C, H, W;
    std::vector<std::vector<std::vector<float>>> data;

    ConvMap() : C(0), H(0), W(0) {}

    ConvMap(int c, int h, int w)
        : C(c), H(h), W(w),
        data(c, std::vector<std::vector<float>>(h, std::vector<float>(w, 0.0f))) {
    }
};

// ================= 前向：卷积 =================
inline ConvMap conv(
    const ConvMap& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& kernels,
    const std::vector<float>& biases,
    int stride,
    int pad
) {
    int IC = input.C;
    int KH = (int)kernels[0][0].size();
    int KW = (int)kernels[0][0][0].size();
    int OC = (int)kernels.size();

    int outH = (input.H + 2 * pad - KH) / stride + 1;
    int outW = (input.W + 2 * pad - KW) / stride + 1;

    ConvMap output(OC, outH, outW);

    for (int IOC = 0; IOC < OC; ++IOC)
        for (int IOH = 0; IOH < outH; ++IOH)
            for (int IOW = 0; IOW < outW; ++IOW) {
                float sum = biases[IOC];

                for (int IIC = 0; IIC < IC; ++IIC)
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = IOH * stride + kh - pad;
                            int iw = IOW * stride + kw - pad;
                            if (ih < 0 || ih >= input.H) continue;
                            if (iw < 0 || iw >= input.W) continue;

                            sum += input.data[IIC][ih][iw] * kernels[IOC][IIC][kh][kw];
                        }

                output.data[IOC][IOH][IOW] = sum;
            }
    return output;
}

// ================= 前向：ReLU =================
inline ConvMap relu(const ConvMap& input) {
    ConvMap output(input.C, input.H, input.W);
    for (int c = 0; c < input.C; ++c)
        for (int h = 0; h < input.H; ++h)
            for (int w = 0; w < input.W; ++w) {
                float v = input.data[c][h][w];
                output.data[c][h][w] = (v < 0.0f) ? 0.0f : v;
            }
    return output;
}

// ================= 前向：池化 =================
inline ConvMap Pool2x2(const ConvMap& input) {
    int outH = input.H / 2;
    int outW = input.W / 2;
    ConvMap output(input.C, outH, outW);

    for (int c = 0; c < input.C; ++c)
        for (int oh = 0; oh < outH; ++oh)
            for (int ow = 0; ow < outW; ++ow) {
                int ih = oh * 2;
                int iw = ow * 2;

                float m = input.data[c][ih][iw];
                if (input.data[c][ih][iw + 1] > m) m = input.data[c][ih][iw + 1];
                if (input.data[c][ih + 1][iw] > m) m = input.data[c][ih + 1][iw];
                if (input.data[c][ih + 1][iw + 1] > m) m = input.data[c][ih + 1][iw + 1];

                output.data[c][oh][ow] = m;
            }
    return output;
}
