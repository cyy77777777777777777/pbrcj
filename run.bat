@echo off
chcp 65001 > nul
echo 启动 AI PBR 贴图生成器...
python texture_generator.py
pause
