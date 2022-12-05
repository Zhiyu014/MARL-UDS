call where conda.bat
call conda activate storm
cd /d %~dp0
python train_chaohu.py
pause