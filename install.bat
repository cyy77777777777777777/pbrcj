@echo off
chcp 65001 > nul
echo ============================================
echo   AI PBR 贴图生成器 — 安装脚本
echo ============================================
echo.

:: 检查 Python 是否安装
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或更高版本。
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 升级 pip ...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [2/3] 安装 PyTorch (CUDA 版)...
echo.
echo ----------------------------------------------------------------
echo  PyTorch CUDA wheel 需从官方服务器下载 (~2.5 GB)，国内速度较慢。
echo  强烈建议先用 IDM / Motrix / 浏览器手动下载：
echo.
echo  https://download.pytorch.org/whl/cu124/torch-2.6.0%%2Bcu124-cp311-cp311-win_amd64.whl
echo  https://download.pytorch.org/whl/cu124/torchvision-0.21.0%%2Bcu124-cp311-cp311-win_amd64.whl
echo.
echo  下载完成后将两个 .whl 文件放到本目录，重新运行此脚本即可离线安装。
echo ----------------------------------------------------------------
echo.

:: 检查本目录是否有离线 wheel
set TORCH_WHL=
set TORCHVISION_WHL=
for %%f in (torch-*.whl) do set TORCH_WHL=%%f
for %%f in (torchvision-*.whl) do set TORCHVISION_WHL=%%f

if defined TORCH_WHL (
    echo 检测到本地 wheel 文件，使用离线安装...
    python -m pip install %TORCH_WHL% %TORCHVISION_WHL%
) else (
    echo 未检测到本地 CUDA wheel，默认安装 CPU 版 PyTorch（最快可用）...
    python -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    echo 如需 GPU 加速，请先手动下载上方两个 cu124 wheel 到本目录后重跑本脚本。
)

echo.
echo [3/3] 安装其余依赖（清华镜像）...
python -m pip install transformers accelerate huggingface_hub controlnet-aux opencv-python scipy gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ============================================
echo   安装完成！
echo   运行方式: python texture_generator.py
echo   或双击    run.bat
echo ============================================
echo.
pause
