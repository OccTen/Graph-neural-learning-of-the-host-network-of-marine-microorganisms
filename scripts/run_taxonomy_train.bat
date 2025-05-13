@echo off
echo 正在运行分类学图GraphSAGE训练脚本...
call conda activate graphsage_env
python training/train_taxonomy_graphsage.py
pause 