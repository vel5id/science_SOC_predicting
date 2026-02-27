@echo off
echo Создание целевой директории в WSL...
wsl bash -c "mkdir -p ~/projects/science-article"

echo Копирование файлов проекта...
wsl bash -c "cp -a \"$(wslpath -u '%CD%')\"/. ~/projects/science-article/"

echo Запуск Antigravity...
wsl bash -c "cd ~/projects/science-article && antigravity ."

pause