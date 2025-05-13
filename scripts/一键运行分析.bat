@echo off
chcp 65001 > nul
title 分类学特征推断与可视化分析

SETLOCAL EnableDelayedExpansion

REM 切换到项目根目录
cd /d "%~dp0\.."

echo ================================================
echo       分类学特征推断与可视化分析 - 一键运行
echo ================================================
echo.

REM 配置参数（默认值）
set MODEL_PATH=weights\taxonomy_graphsage_optimized.pt
set TEST_DATA=data\Independent test set.xlsx
set ENCODERS=data\taxonomy_values.json
set PREDICTION_OUTPUT=output\taxonomy_predictions.json
set VISUALIZE_OUTPUT=output\interaction_analysis

REM 确保输出目录存在
if not exist output mkdir output

REM 检查模型文件是否存在
if not exist "%MODEL_PATH%" (
  echo [警告] 未找到默认模型文件: %MODEL_PATH%
  echo 正在查找可用的模型文件...
  
  REM 尝试查找其他模型文件
  for %%F in (weights\*.pt) do (
    echo 找到模型文件: %%F
    set MODEL_PATH=%%F
    goto :FOUND_MODEL
  )
  
  echo [警告] 未找到任何模型文件，将使用默认路径继续
  
  :FOUND_MODEL
)

REM 检查测试数据文件是否存在
if not exist "%TEST_DATA%" (
  echo [警告] 未找到默认测试数据文件: %TEST_DATA%
  echo 正在查找可用的测试数据文件...
  
  REM 尝试查找Excel文件
  for %%F in (data\*.xlsx) do (
    echo 找到测试数据: %%F
    set TEST_DATA=%%F
    goto :FOUND_TEST_DATA
  )
  
  echo [警告] 未找到任何测试数据文件，将使用默认路径继续
  
  :FOUND_TEST_DATA
)

REM 检查编码器文件是否存在
if not exist "%ENCODERS%" (
  echo [警告] 未找到编码器文件: %ENCODERS%
  echo 正在查找可用的编码器文件...
  
  REM 尝试查找JSON文件
  for %%F in (data\*.json) do (
    echo 找到编码器文件: %%F
    set ENCODERS=%%F
    goto :FOUND_ENCODERS
  )
  
  echo [警告] 未找到任何编码器文件，将使用默认路径继续
  
  :FOUND_ENCODERS
)

REM 确保可视化目录存在
if not exist "%VISUALIZE_OUTPUT%" mkdir "%VISUALIZE_OUTPUT%"

echo [第1步] 运行分类学特征推断...
echo.
echo 模型路径: %MODEL_PATH%
echo 测试数据: %TEST_DATA%
echo 编码器: %ENCODERS%
echo 输出文件: %PREDICTION_OUTPUT%
echo.

REM 运行推断脚本
python scripts\inference_taxonomy.py --model_path "%MODEL_PATH%" --test_data "%TEST_DATA%" --encoders "%ENCODERS%" --output "%PREDICTION_OUTPUT%" --debug

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo [错误] 推断过程失败，错误代码: %ERRORLEVEL%
  echo 无法继续后续步骤。
  goto :ERROR
)

echo.
echo [第1步完成] 推断结果已保存到: %PREDICTION_OUTPUT%
echo.

echo [第2步] 运行可视化分析...
echo.
echo 输入文件: %PREDICTION_OUTPUT%
echo 输出目录: %VISUALIZE_OUTPUT%
echo.

REM 运行可视化脚本
python wrapper\visualization_starter.py --input "%PREDICTION_OUTPUT%" --output "%VISUALIZE_OUTPUT%"

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo [错误] 可视化分析失败，错误代码: %ERRORLEVEL%
  goto :ERROR
)

echo.
echo [第2步完成] 可视化分析已保存到: %VISUALIZE_OUTPUT%
echo.

echo ================================================
echo               全流程已成功完成！
echo ================================================
echo.
echo 推断结果: %PREDICTION_OUTPUT%
echo 可视化结果: %VISUALIZE_OUTPUT%
echo.
echo 已自动打开可视化结果目录。
goto :END

:ERROR
echo.
echo ================================================
echo                 处理过程出错
echo ================================================
echo 请查看上方错误信息并解决问题后再次运行。

:END
pause 