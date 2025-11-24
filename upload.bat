@echo off
cd /d "%~dp0"

echo ===============================
echo  Auto Git Upload to Nano_shape
echo ===============================

echo Checking current branch...
git branch

echo Switching to Nano_shape branch...
git checkout Nano_shape

echo Adding all files...
git add .

echo Commit message:
set /p msg="Enter commit message: "

git commit -m "%msg%"

echo Pushing to remote...
git push origin Nano_shape

echo ===============================
echo   Upload Completed Successfully!
echo ===============================
pause
