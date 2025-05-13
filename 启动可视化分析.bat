@echo off
chcp 65001 > nul
title 微生物-宿主互作可视化分析

REM 切换到批处理文件所在目录
cd /d "%~dp0"

echo 正在启动微生物与宿主互作可视化分析...
echo.

REM 检查Python环境
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo 错误: 未检测到Python环境，请确保已安装Python并添加到PATH
  echo.
  pause
  exit /b 1
)

REM 检查预测结果文件是否存在
set PREDICTION_FILE=output\taxonomy_predictions.json
if not exist "%PREDICTION_FILE%" (
  echo 警告: 未找到默认预测结果文件: %PREDICTION_FILE%
  echo 正在查找其他可能的预测结果文件...
  
  REM 尝试查找其他预测结果文件
  for %%F in (output\*.json) do (
    echo 找到预测结果文件: %%F
    set PREDICTION_FILE=%%F
    goto :FOUND_PREDICTION
  )
  
  echo 警告: 未找到任何预测结果文件
  echo 请先运行推断脚本生成预测结果，或手动指定文件路径
  goto :RUN_WITHOUT_FILE
  
  :FOUND_PREDICTION
)

echo 使用预测结果文件: %PREDICTION_FILE%
echo.

:RUN_WITHOUT_FILE
REM 运行可视化启动器脚本
python wrapper\visualization_starter.py

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo 可视化分析运行出错，请查看上方错误信息
  echo.
  echo 可能是缺少必要的Python库，请执行以下命令安装：
  echo pip install matplotlib seaborn networkx scikit-learn pyvis
  echo.
)

pause 