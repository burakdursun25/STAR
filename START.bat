@echo off
chcp 65001 >nul
title STAR-1 Launcher
set PYTHONIOENCODING=utf-8

set "BLENDER=C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
set "SCENE=%~dp0StarScene.blend"
set "SETUP=%~dp0blender\auto_setup.py"
set "STAR_DIR=%~dp0"

echo.
echo  +===========================================+
echo  ^|       STAR-1 LIVE POSE SYSTEM             ^|
echo  +===========================================+
echo.

:: -- Adim 1: Blender kurulumu (headless, sadece ilk seferde) --
if "%1"=="--setup" (
    echo  [KURULUM] Blender addon kuruluyor...
    "%BLENDER%" "%SCENE%" --python "%SETUP%" --background
    if errorlevel 1 (
        echo.
        echo  [HATA] Blender kurulumu basarisiz!
        pause
        exit /b 1
    )
    echo  [KURULUM] Tamamlandi!
    echo.
)

:: -- Blender varligi kontrol --
if not exist "%BLENDER%" (
    echo  [HATA] Blender 5.0 bulunamadi!
    echo  Beklenen konum: %BLENDER%
    echo.
    pause
    exit /b 1
)

:: -- StarScene.blend varligi kontrol --
if not exist "%SCENE%" (
    echo  [HATA] StarScene.blend bulunamadi!
    echo  Beklenen konum: %SCENE%
    echo.
    pause
    exit /b 1
)

:: -- Python kontrol --
python --version >nul 2>&1
if errorlevel 1 (
    echo  [HATA] Python bulunamadi! Python 3.10+ yuklu olmali.
    echo.
    pause
    exit /b 1
)

:: -- Adim 2: Python star_live arka planda basalt --
echo  [1/2] Python kamera akisi baslatiliyor (port 7777)...
set "HELPER=%~dp0python_start.bat"
start "STAR-1 Python" cmd /k "%HELPER%"

:: Blender'in porta baglanabilmesi icin 3 saniye bekle
timeout /t 3 /nobreak >nul

:: -- Adim 3: Blender'i ac --
echo  [2/2] Blender 5.0 aciliyor...
echo.
echo  +------------------------------------------+
echo  ^|  Blender actiktan sonra:                  ^|
echo  ^|  N tusu - STAR Live - Dinlemeyi Baslat    ^|
echo  +------------------------------------------+
echo.
"%BLENDER%" "%SCENE%"
