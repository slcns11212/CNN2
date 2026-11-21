#pragma once
// ============================================================================
// file.h —— 文件系统工具 + acm 二进制模型格式(header-only)
//
// 阶段一(本文件)只提供两类能力，供后续 T3/T4 的 save/load 分流复用：
//   T1) 路径 / 目录 / 格式探测工具
//   T2) acm 固定 1024 字节头部的编解码 + 数据区 CRC32
//
// 约定：
//   * 统一使用 Win32 API（CreateDirectoryA / GetFileAttributesA），
//     不使用 <filesystem>（Debug 配置为 C++14，无法使用）。
//   * 头部按字节手动小端写入/读取（不依赖 struct padding/主机字节序），
//     确保 Debug/Release、32/64 位下逐字节一致。
//   * header-only：函数均以 inline 定义，无全局可变状态依赖，可独立编译。
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>

#include "net_spec.h"

// ============================================================================
// T1：路径 / 目录 / 格式探测工具
// ============================================================================

// 模型文件格式（探测结果）。
//   Txt     : 旧文本格式
//   Acm     : 新二进制格式（魔数 "ACMN" 命中）
//   Unknown : 显式 .acm 后缀但魔数不符（= 损坏），由调用方报错
enum class ModelFormat { Txt, Acm, Unknown };

// 取路径最后一段（basename），同时识别 '/' 与 '\\'（迁移自 main.cpp baseName）。
inline std::string pathBaseName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// 取“去扩展名”的文件名主干（迁移自 main.cpp modelBaseName）：
//   E:\CNN\models\b4_1.txt -> "b4_1"；无扩展名保持原样；结果为空回退 "model"。
inline std::string pathStem(const std::string& path) {
    std::string n = pathBaseName(path);
    size_t dot = n.find_last_of('.');
    if (dot != std::string::npos) n = n.substr(0, dot);
    return n.empty() ? std::string("model") : n;
}

// 返回小写扩展名（含点，如 ".txt"）；无扩展名返回空串。
// 仅在最后一个路径分隔符之后查找 '.'，避免命中目录名中的点。
inline std::string pathExtLower(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string();
    if (sep != std::string::npos && dot < sep) return std::string();
    std::string ext = path.substr(dot);
    for (char& c : ext) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return ext;
}

// 判断文件/目录是否存在（Win32）。
inline bool fileExists(const std::string& path) {
    if (path.empty()) return false;
    DWORD a = GetFileAttributesA(path.c_str());
    return (a != INVALID_FILE_ATTRIBUTES);
}

// 逐段创建目录（迁移自 main.cpp ensureDirectory）：
//   处理盘符 "X:"；ERROR_ALREADY_EXISTS 视为成功；'/' 统一为 '\\'。
inline bool ensureDir(const std::string& dir) {
    if (dir.empty()) return false;
    std::string d = dir;
    for (char& c : d) if (c == '/') c = '\\';   // 统一分隔符
    std::string cur;
    size_t i = 0;
    if (d.size() >= 2 && d[1] == ':') { cur = d.substr(0, 2); i = 2; }   // 盘符前缀
    while (i < d.size()) {
        if (d[i] == '\\') { ++i; continue; }
        size_t j = i;
        while (j < d.size() && d[j] != '\\') ++j;
        std::string seg = d.substr(i, j - i);
        if (cur.empty()) cur = seg;
        else cur += (cur.back() == '\\') ? seg : ("\\" + seg);
        // 跳过纯盘符段（"X:"）；其余逐段创建
        if (cur.back() != ':') {
            if (!CreateDirectoryA(cur.c_str(), nullptr)) {
                DWORD e = GetLastError();
                if (e != ERROR_ALREADY_EXISTS) return false;
            }
        }
        i = j;
    }
    return true;
}

// 格式探测：
//   1) 读前 4 字节 == "ACMN" -> Acm（不论后缀）
//   2) 否则 ext == ".acm"    -> Unknown（.acm 但魔数不符 = 损坏，调用方报错）
//   3) ext == ".txt"         -> Txt
//   4) 无 / 未知后缀         -> 尝试按 Txt（兼容旧文本），由调用方再决定
inline ModelFormat probeModelFormat(const std::string& path) {
    // 1) 优先魔数
    {
        std::ifstream fin(path, std::ios::binary);
        if (fin) {
            char magic[4] = { 0, 0, 0, 0 };
            fin.read(magic, 4);
            if (fin.gcount() == 4 &&
                magic[0] == 'A' && magic[1] == 'C' && magic[2] == 'M' && magic[3] == 'N')
                return ModelFormat::Acm;
        }
    }
    // 2/3) 回退扩展名
    const std::string ext = pathExtLower(path);
    if (ext == ".acm") return ModelFormat::Unknown;   // 魔数不符：损坏
    if (ext == ".txt") return ModelFormat::Txt;
    // 4) 无/未知后缀：按文本尝试
    return ModelFormat::Txt;
}

// 规范化模型后缀：Acm -> ".acm"，Txt -> ".txt"（替换已有扩展名，无扩展名则补）。
// Unknown 或其它格式原样返回。
inline std::string withModelExt(const std::string& path, ModelFormat f) {
    const char* ext = nullptr;
    if (f == ModelFormat::Acm)      ext = ".acm";
    else if (f == ModelFormat::Txt) ext = ".txt";
    if (!ext) return path;

    size_t sep = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep))
        return path.substr(0, dot) + ext;   // 替换末尾扩展名
    return path + ext;                       // 无扩展名：追加
}

// ============================================================================
// T2：acm 固定头部编解码（恒 1024 字节，小端）
// ============================================================================
//
// 布局（总大小 = 1024 + total*4 字节）：
//   [0,   64)   标量区
//   [64,  320)  #SIG 字符串区（256B，逐字符与 txt 首行一致，NUL 填充）
//   [320, 800)  层表区（480B = 24 条 × 20B，有效条目数 = depth）
//   [800, 1024) 保留区（224B，零填充）
//   [1024, ...) 数据区：total × float32（顺序 = NetSpec 扁平布局，与 flattenParams 一致）

static constexpr int ACM_HEADER_SIZE = 1024;   // 固定头部字节数
static constexpr int ACM_SIG_OFF     = 64;     // #SIG 区起始
static constexpr int ACM_SIG_SIZE    = 256;    // #SIG 区大小
static constexpr int ACM_LAYER_OFF   = 320;    // 层表区起始
static constexpr int ACM_LAYER_SIZE  = 20;     // 单条层记录字节数
static constexpr int ACM_RESERVED_OFF = 800;   // 保留区起始
static constexpr char ACM_MAGIC[5] = "ACMN";   // 魔数（含结尾 NUL）

// acm 头部标志位
static constexpr uint16_t ACM_FLAG_CRC32 = 0x0001;   // bit0：数据区含 CRC32
static constexpr uint16_t ACM_FLAG_SIG   = 0x0002;   // bit1：含 #SIG 串

// 精度编码：0=FP32, 1=FP16, 2=INT8（当前仅实现 FP32）
static constexpr uint8_t ACM_PREC_FP32 = 0;

// 单条层记录（对应 20B 磁盘记录；结构体本身不参与磁盘布局，仅作解码产物）。
struct AcmLayerEntry {
    int16_t  outC    = 0;
    int16_t  inC     = 0;
    int16_t  kH      = 0;
    int16_t  kW      = 0;
    int16_t  stride  = 1;
    int16_t  pad     = 0;
    uint8_t  pool    = 2;
    uint8_t  bn      = 0;
    uint8_t  res     = 0;
    uint8_t  resPos  = 0;
    int16_t  resFrom = -1;
    uint16_t pad2    = 0;   // 保留，恒 0
};

// 解码后的 acm 头部（内存表示；磁盘布局由 encode/decode 手动按字节处理）。
struct AcmHeader {
    char     magic[4]       = { 'A', 'C', 'M', 'N' };
    uint16_t headerVersion  = 1;
    uint8_t  formatVersion  = 1;
    uint8_t  precision      = 0;
    uint16_t flags          = 0;
    uint8_t  bn             = 0;
    uint8_t  depth          = 0;
    int32_t  inH            = 0;
    int32_t  inW            = 0;
    int32_t  classes        = 0;
    int32_t  flat           = 0;
    int32_t  total          = 0;
    uint32_t dataBytes      = 0;
    uint32_t sigLen         = 0;
    uint32_t crc32          = 0;
    char     reserved0[20]  = { 0 };
    char     sig[ACM_SIG_SIZE] = { 0 };
    AcmLayerEntry layers[MAX_DEPTH];
};

// ---- 小端读写辅助（手动按字节，避免 padding/主机字节序歧义） ----
inline void acmPutU16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}
inline void acmPutU32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}
inline void acmPutI32(uint8_t* p, int32_t v) { acmPutU32(p, (uint32_t)v); }

inline uint16_t acmGetU16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
inline uint32_t acmGetU32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline int32_t acmGetI32(const uint8_t* p) { return (int32_t)acmGetU32(p); }

// 从 #SIG 串解析格式版本 ver（存在 "ver=2"/"ver:2" 时返回 2，否则 1）。
// 与 main.cpp makeSignature 的写法保持一致（大小写不敏感）。
inline int acmSigVersion(const std::string& sig) {
    for (size_t i = 0; i + 3 < sig.size(); ++i) {
        char c0 = sig[i], c1 = sig[i + 1], c2 = sig[i + 2];
        if (c0 == 'v' || c0 == 'V') {
            if ((c1 == 'e' || c1 == 'E') && (c2 == 'r' || c2 == 'R')) {
                size_t k = i + 3;
                if (k < sig.size() && (sig[k] == '=' || sig[k] == ':')) {
                    ++k;
                    // 跳过前导空格
                    while (k < sig.size() && (sig[k] == ' ' || sig[k] == '\t')) ++k;
                    if (k < sig.size() && sig[k] >= '0' && sig[k] <= '9') {
                        int v = 0;
                        while (k < sig.size() && sig[k] >= '0' && sig[k] <= '9') {
                            v = v * 10 + (sig[k] - '0');
                            if (v > 100000) break;
                            ++k;
                        }
                        if (v >= 2) return v;
                    }
                }
            }
        }
    }
    return 1;
}

// CRC32（IEEE 802.3，多字节反序，多项式 0xEDB88320；与常见 zlib/gzip 一致）。
inline uint32_t acmCrc32(const uint8_t* data, size_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= (uint32_t)data[i];
        for (int k = 0; k < 8; ++k) {
            uint32_t mask = (uint32_t)0u - (crc & 1u);   // 全 1 / 全 0
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

// 用 NetSpec + sig 串 + precision 编码出 1024B 头（out 被重置为恰好 ACM_HEADER_SIZE 字节）。
//   * precision 参数保留以供将来扩展；当前仅实现 FP32，故恒写 precision=0。
//   * flags 当前仅按“含 sig 串”置 bit1；数据区 CRC32 由调用方（T3）在得到数据后
//     调用 acmCrc32 写回偏移 40 并置 bit0（本函数此时 crc32 字段写 0）。
//   * 成功返回 true，out.size()==1024；失败返回 false 并写 err。
inline bool encodeAcmHeader(const NetSpec& spec, const std::string& sig, uint8_t precision,
                            std::vector<uint8_t>& out, std::string* err) {
    (void)precision;   // 占位：当前恒写 FP32(0)

    if (spec.depth < 1 || spec.depth > MAX_DEPTH) {
        if (err) *err = "acm 编码失败：depth 越界";
        return false;
    }

    out.assign(ACM_HEADER_SIZE, 0);   // 全零起步，保留区/填充天然为零
    uint8_t* p = out.data();

    // R5：#SIG 实际写入字节数（截断到 ACM_SIG_SIZE-1 = 255，保留 1 字节 NUL 终止）。
    //     sigLen 必须写这个截断后的值，否则超长签名会让 read/加载按 sigLen 读入尾随 NUL。
    size_t sigN = sig.size();
    if (sigN > (size_t)(ACM_SIG_SIZE - 1)) sigN = (size_t)(ACM_SIG_SIZE - 1);

    // ---- 标量区 ----
    std::memcpy(p + 0, ACM_MAGIC, 4);          // magic "ACMN"
    acmPutU16(p + 4, 1);                       // headerVersion = 1
    p[6] = (uint8_t)acmSigVersion(sig);        // formatVersion = #SIG 的 ver（1/2）
    p[7] = ACM_PREC_FP32;                      // precision（占位，恒 0）
    uint16_t flags = 0;
    if (!sig.empty()) flags |= ACM_FLAG_SIG;   // bit1：含 sig 串
    acmPutU16(p + 8, flags);                   // flags（bit0 CRC32 由 T3 补）
    p[10] = (uint8_t)(spec.bn ? 1 : 0);        // bn
    p[11] = (uint8_t)spec.depth;               // depth
    acmPutI32(p + 12, spec.inH);               // inH
    acmPutI32(p + 16, spec.inW);               // inW
    acmPutI32(p + 20, spec.classes);           // classes
    acmPutI32(p + 24, spec.flat);              // flat
    acmPutI32(p + 28, spec.total);             // total
    acmPutU32(p + 32, (uint32_t)spec.total * 4u);   // dataBytes = total*4
    acmPutU32(p + 36, (uint32_t)sigN);              // sigLen（截断后实际字节数，见 R5）
    acmPutU32(p + 40, 0);                      // crc32（由 T3 计算写回）
    // reserved0[20] 已为零

    // ---- #SIG 字符串区（NUL 填充，保留 1 字节 NUL 终止） ----
    {
        std::memcpy(p + ACM_SIG_OFF, sig.data(), sigN);
    }

    // ---- 层表区（有效条目数 = depth，其余保持零） ----
    for (int i = 0; i < spec.depth; ++i) {
        uint8_t* e = p + ACM_LAYER_OFF + (size_t)i * ACM_LAYER_SIZE;
        const LayerSpec& L = spec.layer[i];
        acmPutU16(e + 0,  (uint16_t)(int16_t)L.outC);
        acmPutU16(e + 2,  (uint16_t)(int16_t)L.inC);
        acmPutU16(e + 4,  (uint16_t)(int16_t)L.kH);
        acmPutU16(e + 6,  (uint16_t)(int16_t)L.kW);
        acmPutU16(e + 8,  (uint16_t)(int16_t)L.stride);
        acmPutU16(e + 10, (uint16_t)(int16_t)L.pad);
        e[12] = (uint8_t)L.pool;
        e[13] = (uint8_t)(L.bn ? 1 : 0);
        e[14] = (uint8_t)L.res;
        e[15] = (uint8_t)L.resPos;
        acmPutU16(e + 16, (uint16_t)(int16_t)L.resFrom);
        acmPutU16(e + 18, 0);   // pad2
    }

    // [ACM_RESERVED_OFF, 1024) 保持零填充
    return true;
}

// 解码 1024B 头 -> AcmHeader + NetSpec（调 computeNetSpec/validateNetSpec）+ sig 串。
//   * in 至少需含 ACM_HEADER_SIZE 字节（其后数据区长度由 total 决定，本函数不校验）。
//   * precision != 0 时返回 false 并在 err 写明“暂不支持”。
//   * 成功返回 true；失败返回 false 并写 err（若 err 非空）。
inline bool decodeAcmHeader(const std::vector<uint8_t>& in, AcmHeader& h, NetSpec& spec,
                            std::string& sig, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    if (in.size() < (size_t)ACM_HEADER_SIZE)
        return fail("acm 头部不完整（不足 1024 字节）");

    const uint8_t* p = in.data();
    if (std::memcmp(p + 0, ACM_MAGIC, 4) != 0)
        return fail("acm 魔数不符（缺少 \"ACMN\"）");

    const uint16_t hver = acmGetU16(p + 4);
    if (hver != 1) return fail("acm 头部版本不支持（headerVersion=" + std::to_string(hver) + "）");

    const uint8_t prec = p[7];
    if (prec != ACM_PREC_FP32)
        return fail("acm 精度 " + std::to_string((int)prec) + " 暂不支持（当前仅 fp32）");

    const uint8_t depth = p[11];
    if (depth < 1 || depth > MAX_DEPTH)
        return fail("acm depth 越界（" + std::to_string((int)depth) + "）");

    // ---- 标量区 ----
    std::memcpy(h.magic, p + 0, 4);
    h.headerVersion = hver;
    h.formatVersion = p[6];
    h.precision     = prec;
    h.flags         = acmGetU16(p + 8);
    h.bn            = p[10];
    h.depth         = depth;
    h.inH           = acmGetI32(p + 12);
    h.inW           = acmGetI32(p + 16);
    h.classes       = acmGetI32(p + 20);
    h.flat          = acmGetI32(p + 24);
    h.total         = acmGetI32(p + 28);
    h.dataBytes     = acmGetU32(p + 32);
    h.sigLen        = acmGetU32(p + 36);
    h.crc32         = acmGetU32(p + 40);
    std::memcpy(h.reserved0, p + 44, 20);

    // ---- #SIG 串（按 sigLen，夹到 256 上限） ----
    {
        size_t n = (size_t)h.sigLen;
        if (n > (size_t)ACM_SIG_SIZE) n = (size_t)ACM_SIG_SIZE;
        std::memcpy(h.sig, p + ACM_SIG_OFF, n);
        if (n < (size_t)ACM_SIG_SIZE) h.sig[n] = '\0';
        sig.assign((const char*)(p + ACM_SIG_OFF), n);
    }

    // ---- 层表区 ----
    for (int i = 0; i < depth; ++i) {
        const uint8_t* e = p + ACM_LAYER_OFF + (size_t)i * ACM_LAYER_SIZE;
        AcmLayerEntry& le = h.layers[i];
        le.outC    = (int16_t)acmGetU16(e + 0);
        le.inC     = (int16_t)acmGetU16(e + 2);
        le.kH      = (int16_t)acmGetU16(e + 4);
        le.kW      = (int16_t)acmGetU16(e + 6);
        le.stride  = (int16_t)acmGetU16(e + 8);
        le.pad     = (int16_t)acmGetU16(e + 10);
        le.pool    = e[12];
        le.bn      = e[13];
        le.res     = e[14];
        le.resPos  = e[15];
        le.resFrom = (int16_t)acmGetU16(e + 16);
        le.pad2    = acmGetU16(e + 18);
    }

    // ---- 还原 NetSpec 结构并复算/校验 ----
    spec = NetSpec();
    spec.inH     = h.inH;
    spec.inW     = h.inW;
    spec.classes = h.classes;
    spec.depth   = depth;
    spec.bn      = (h.bn != 0);
    for (int i = 0; i < depth; ++i) {
        const AcmLayerEntry& le = h.layers[i];
        LayerSpec& L = spec.layer[i];
        L.outC    = le.outC;
        L.inC     = le.inC;
        L.kH      = le.kH;
        L.kW      = le.kW;
        L.bn      = le.bn ? 1 : 0;
        L.stride  = le.stride;
        L.pad     = le.pad;
        L.pool    = le.pool;
        L.res     = le.res;
        L.resFrom = le.resFrom;
        L.resPos  = le.resPos;
    }
    computeNetSpec(spec);   // 复算 H/W 与各段偏移

    std::string verr;
    if (!validateNetSpec(spec, &verr))
        return fail("acm 规格非法：" + verr);

    return true;
}
