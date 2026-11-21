==================================================
 CNN2 免安装 CUDA 版  部署说明（工业机 P100）
==================================================

一、这是什么
  本目录是自包含的 CNN2 可执行程序，已内置 NVIDIA GPU（CUDA sm_60/sm_61）机器码，
  可直接在装有 NVIDIA 驱动的工业机上运行（目标机驱动 582.53，P100）。
  无需安装 CUDA Toolkit，也无需单独安装 Visual C++ 运行库。

二、目录内容（务必保持所有文件在同一目录）
  CNN2.exe            主程序
  cudart64_12.dll     CUDA 运行时（必须与 exe 同目录）
  msvcp140.dll        MSVC C++ 运行库
  vcruntime140.dll    MSVC C 运行库
  vcruntime140_1.dll  MSVC C 运行库（异常处理）

  注意：不要只拷贝 CNN2.exe，必须连同 4 个 dll 一起拷贝到工业机同一目录。
  KERNEL32.dll 以及 api-ms-win-crt-*.dll 为 Windows 系统自带，无需拷贝。

三、运行前提
  1. Windows 10 及以上（64 位）。
  2. 已安装 NVIDIA 显卡驱动（支持 P100），系统能正常识别 GPU。
  3. 无需 CUDA Toolkit，无需 VC++ Redist。

四、数据格式
  - 数据放在一个目录中，程序读取该目录下所有 *.txt 文件（不递归子目录）。
  - 每个 txt = 一张图：32 行，每行 32 个字符，字符只能是 '0' 或 '1'（0=背景，1=笔画）。
  - 类别标签 = 文件名第一个字符（'0'~'9'）。
    例如：3_a.txt 表示标签 3；07_1.txt 表示标签 0。
  - 图像必须为 32x32，行数或每行字符数不符会报错。

五、用法示例（在 cmd 中先进入本目录，再执行；下面把程序写作 CNN2.exe）

  1) 训练（bkc）
     CNN2.exe <训练目录> bkc --gpu --epochs 50 --batch 128
     - 若同目录已存在 model.txt，会先加载再继续训练；不存在则随机初始化。
     - --cpu 可强制使用 CPU；--batch 默认 128。
     - 训练结果保存到 --model 指定文件（默认 model.txt），
       同时每个 epoch 输出 model_epochN.txt。

  2) 混合训练（阶段D：两套数据按 (域,类别) 分层均衡）
     CNN2.exe <主目录> bkc --data2 <第二套目录> --gpu --epochs 50

  3) 数据增强（阶段E，默认关闭）
     CNN2.exe <训练目录> bkc --gpu --aug --aug_k 2 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1
     - --aug_elastic 4   弹性形变强度（默认 0 关闭，建议从 4 起）
     - --aug_elastic_sigma 4.5   弹性位移场高斯 σ
     - --aug_thick 50    笔画粗细增强触发概率 %（默认 50，0 关闭）

  4) 测试（对整个目录，打印精度）
     CNN2.exe <测试目录> test --model model.txt

  5) 单张推理
     CNN2.exe <图片.txt> --model model.txt

  6) 生成随机初始模型
     CNN2.exe <路径> gen --model model.txt

六、预处理参数（训练/测试/推理共用，务必保持一致）
  --prep crop,deskew,resize,center,thick   启用步骤（逗号分隔）；none 关闭全部
  --prep_target N         resize 目标框最长边（16-30，默认 28）
  --prep_stretch          拉伸为 target×target（否则保持长宽比）
  --prep_ink X            目标墨迹比（thick 用，默认 0.20，0=关闭）
  --prep_deskew_max D     deskew 角度限幅（度，默认 15）
  --diag                  打印预处理前后统计（墨迹比/质心/包围盒/二值断言）

  重要提示：训练、测试、推理必须使用完全相同的预处理配置，
  否则识别精度会明显下降。

七、常见问题
  - 提示缺少 dll：确认 CNN2.exe 与 4 个 dll 在同一目录。
  - GPU 不可用：确认 NVIDIA 驱动已安装且显卡被系统识别；可用 --cpu 回退。
  - batch=1 时 GPU 与 CPU 结果在浮点容差内一致；默认 batch=128。

八、版本信息
  编译环境：MSVC v143 (14.44)，CUDA 12.9；
  exe 含 sm_60(P100) 与 sm_61 机器码，可直接在 P100 上运行。
