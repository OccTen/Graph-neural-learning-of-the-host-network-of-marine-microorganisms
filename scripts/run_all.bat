@echo off
REM 全流程批处理文件 - 运行推断和分析

SETLOCAL

REM 切换到项目根目录（假设脚本在scripts目录）
cd /d "%~dp0\.."

echo ================================================
echo             分类学特征推断系统 - 全流程
echo ================================================
echo.

REM 配置参数
set MODEL_PATH=weights\taxonomy_graphsage_optimized.pt
set TEST_DATA=data\Independent test set.xlsx
set ENCODERS=data\taxonomy_values.json
set PREDICTION_OUTPUT=output\taxonomy_predictions.json
set ANALYSIS_OUTPUT=output\relationship_analysis.html

REM 确保输出目录存在
if not exist output mkdir output

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

echo [第2步] 运行噬菌体-宿主关系分析...
echo.
echo 输入文件: %PREDICTION_OUTPUT%
echo 输出报告: %ANALYSIS_OUTPUT%
echo.

REM 运行分析脚本
python scripts\analyze_phage_host_relationships.py --input "%PREDICTION_OUTPUT%" --output "%ANALYSIS_OUTPUT%" --visualize

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo [错误] 分析过程失败，错误代码: %ERRORLEVEL%
  goto :ERROR
)

echo.
echo [第2步完成] 分析报告已保存到: %ANALYSIS_OUTPUT%
echo.

echo ================================================
echo               全流程已成功完成！
echo ================================================
echo.
echo 推断结果: %PREDICTION_OUTPUT%
echo 分析报告: %ANALYSIS_OUTPUT%
echo 可视化图表: output目录
echo.
echo 可以在浏览器中打开分析报告查看详细内容。
goto :END

:ERROR
echo.
echo ================================================
echo                 处理过程出错
echo ================================================
echo 请查看上方错误信息并解决问题后再次运行。

:END
pause 