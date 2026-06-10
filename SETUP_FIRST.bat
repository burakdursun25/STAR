@echo off
chcp 65001 >nul
title STAR-1 Temizleme + Kurulum

set "BLENDER=C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
set "SCENE=%~dp0StarScene.blend"
set "SETUP=%~dp0blender\clean_setup.py"

echo.
echo  +===========================================+
echo  ^|    STAR-1 TEMIZLEME + YENIDEN KURULUM    ^|
echo  +===========================================+
echo.
echo  Blender arka planda calistirilıyor...
echo  (Bu islem 20-30 saniye surebilir)
echo.

"%BLENDER%" "%SCENE%" --python "%SETUP%" --background

if errorlevel 1 (
    echo.
    echo  [HATA] Kurulum basarisiz!
    echo.
    pause
    exit /b 1
)

echo.
echo  ================================================
echo  Kurulum tamamlandi!
echo  Simdi START.bat ile sistemi baslatabilirsiniz.
echo  ================================================
echo.
pause
