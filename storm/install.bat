call C:/ProgramData/Anaconda3/Scripts/activate
call conda create --name storm python==3.8
call conda activate storm
cd /d %~dp0
pip install -r requirements.txt
pause