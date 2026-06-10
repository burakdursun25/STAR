@echo off
chcp 65001 >nul
title STAR-1 Python - Kamera Akisi
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo.
echo  [STAR-1] Python kamera akisi baslatiliyor...
echo  Durdurmak icin bu pencereyi kapatabilirsiniz.
echo.

python -m star_live
if errorlevel 1 (
    echo.
    echo  ================================================
    echo  [HATA] star_live basarisiz! Hata mesajina bakin.
    echo  ================================================
    echo.
    pause
)
