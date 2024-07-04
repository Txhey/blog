@echo off

python init.py
git add .
set currentdatetime=%date% %time%
git commit -m "update at : %currentdatetime%"
git push

echo Done!
pause