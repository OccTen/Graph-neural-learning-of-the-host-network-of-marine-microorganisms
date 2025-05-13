@echo off
REM 噬菌体-宿主关系分析批处理文件

SETLOCAL

REM 切换到项目根目录
cd /d "%~dp0\.."

REM 配置参数
set INPUT=output\taxonomy_predictions.json
set OUTPUT=output\relationship_analysis.html

echo 运行噬菌体-宿主关系分析...
echo.
echo 项目根目录: %CD%
echo 输入文件: %INPUT%
echo 输出报告: %OUTPUT%
echo.

REM 运行Python脚本
python scripts\analyze_phage_host_relationships.py --input "%INPUT%" --output "%OUTPUT%" --visualize

echo.
if %ERRORLEVEL% EQU 0 (
  echo 分析成功完成！
  echo 报告已保存到: %OUTPUT%
) else (
  echo 分析过程发生错误，错误代码: %ERRORLEVEL%
)

pause 