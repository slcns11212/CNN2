#pragma once
#include "conv.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

inline std::vector<float> flatten(const ConvMap& p2) {
    std::vector<float> flat;
    for (int c = 0; c < p2.C; ++c)
        for (int h = 0; h < p2.H; ++h)
            for (int w = 0; w < p2.W; ++w)
                flat.push_back(p2.data[c][h][w]);
    return flat;
}

class FullyConnected {
public:
    int inDim;
    int outDim;
    std::vector<std::vector<float>> W;
    std::vector<float> b;

    FullyConnected(int inD, int outD)
        : inDim(inD), outDim(outD),
        W(outD, std::vector<float>(inD, 0.0f)),
        b(outDim, 0.0f) {
    }

    void randomInit() {
        for (int o = 0; o < outDim; ++o)
            for (int i = 0; i < inDim; ++i)
                W[o][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (int o = 0; o < outDim; ++o)
            b[o] = 0.0f;
    }

    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> y(outDim, 0.0f);
        for (int o = 0; o < outDim; ++o) {
            y[o] = b[o];
            for (int i = 0; i < inDim; ++i)
                y[o] += W[o][i] * x[i];
        }
        return y;
    }
};

inline std::vector<float> softmax(const std::vector<float>& z) {
    int n = z.size();
    std::vector<float> p(n);
    float mx = z[0];
    for (int i = 1; i < n; ++i) if (z[i] > mx) mx = z[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { p[i] = std::exp(z[i] - mx); sum += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= sum;
    return p;
}

inline std::vector<float> FC_backward(
    const std::vector<float>& x,
    const std::vector<float>& dy,
    const std::vector<std::vector<float>>& W,
    std::vector<std::vector<float>>& dW,
    std::vector<float>& db,
    std::vector<float>& dx)
{
    int inDim = x.size();
    int outDim = dy.size();

    dW.assign(outDim, std::vector<float>(inDim, 0.0f));
    db.assign(outDim, 0.0f);
    dx.assign(inDim, 0.0f);

    for (int o = 0; o < outDim; ++o) {
        db[o] = dy[o];
        for (int i = 0; i < inDim; ++i) {
            dW[o][i] = dy[o] * x[i];
            dx[i] += dy[o] * W[o][i];
        }
    }
    return dx;
}

inline ConvMap relu_backward(const ConvMap& x, const ConvMap& dy) {
    ConvMap dx(x.C, x.H, x.W);
    for (int c = 0; c < x.C; ++c)
        for (int h = 0; h < x.H; ++h)
            for (int w = 0; w < x.W; ++w)
                dx.data[c][h][w] = (x.data[c][h][w] > 0)
                ? dy.data[c][h][w] : 0.0f;
    return dx;
}

inline ConvMap pool2x2_backward(const ConvMap& x, const ConvMap& dy) {
    ConvMap dx(x.C, x.H, x.W);
    for (int c = 0; c < x.C; ++c)
        for (int oh = 0; oh < dy.H; ++oh)
            for (int ow = 0; ow < dy.W; ++ow) {
                int ih = oh * 2;
                int iw = ow * 2;

                float m = x.data[c][ih][iw];
                int mh = ih, mw = iw;

                if (x.data[c][ih][iw + 1] > m) { m = x.data[c][ih][iw + 1]; mh = ih; mw = iw + 1; }
                if (x.data[c][ih + 1][iw] > m) { m = x.data[c][ih + 1][iw]; mh = ih + 1; mw = iw; }
                if (x.data[c][ih + 1][iw + 1] > m) { m = x.data[c][ih + 1][iw + 1]; mh = ih + 1; mw = iw + 1; }

                dx.data[c][mh][mw] += dy.data[c][oh][ow];
            }
    return dx;
}

inline void conv_backward(
    const ConvMap& x,
    const ConvMap& dy,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& kernels,
    int stride, int pad,
    std::vector<std::vector<std::vector<std::vector<float>>>>& dKernels,
    std::vector<float>& dBiases,
    ConvMap& dx)
{
    int IC = x.C;
    int KH = (int)kernels[0][0].size();
    int KW = (int)kernels[0][0][0].size();
    int OC = (int)kernels.size();

    int outH = dy.H;
    int outW = dy.W;

    dKernels.assign(OC, std::vector<std::vector<std::vector<float>>>(
        IC, std::vector<std::vector<float>>(KH, std::vector<float>(KW, 0.0f))));
    dBiases.assign(OC, 0.0f);
    dx = ConvMap(IC, x.H, x.W);

    for (int oc = 0; oc < OC; ++oc)
        for (int oh = 0; oh < outH; ++oh)
            for (int ow = 0; ow < outW; ++ow) {
                float d = dy.data[oc][oh][ow];
                dBiases[oc] += d;

                for (int ic = 0; ic < IC; ++ic)
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih < 0 || ih >= x.H) continue;
                            if (iw < 0 || iw >= x.W) continue;

                            dKernels[oc][ic][kh][kw] += d * x.data[ic][ih][iw];
                            dx.data[ic][ih][iw] += d * kernels[oc][ic][kh][kw];
                        }
            }
}
