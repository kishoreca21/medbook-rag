@echo off
setlocal enabledelayedexpansion
title Audio Splitter (dated patient naming)

REM ============================================================
REM  Audio Splitter -> one file per patient, named <DATE>_NN
REM  e.g. 06thJun_01.m4a, 06thJun_02.m4a ...
REM  Output folder: <input folder>\<DATE>\
REM  Uses ffmpeg stream copy = instant + lossless.
REM ============================================================

echo.
echo  ===============================================
echo        AUDIO SPLITTER  (dated naming)
echo  ===============================================
echo.

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] ffmpeg not found on PATH.
    echo  Install from https://www.gyan.dev/ffmpeg/builds/ and add bin to PATH.
    pause
    exit /b 1
)

REM --- Input file ---
set "INFILE="
if not "%~1"=="" set "INFILE=%~1"
if "%INFILE%"=="" (
    echo  Tip: you can drag an audio file onto this .bat.
    set /p "INFILE=Enter path to your audio file: "
)
set "INFILE=%INFILE:"=%"
if not exist "%INFILE%" (
    echo  [ERROR] File not found: %INFILE%
    pause
    exit /b 1
)

REM --- Date tag (this becomes the filename prefix AND the folder name) ---
set "DATETAG="
set /p "DATETAG=Enter date tag (e.g. 06thJun): "
if "!DATETAG!"=="" (
    echo  [ERROR] Date tag is required.
    pause
    exit /b 1
)
set "DATETAG=!DATETAG: =!"

REM --- Work out folder and extension of the input ---
for %%F in ("%INFILE%") do (
    set "INDIR=%%~dpF"
    set "EXT=%%~xF"
)

set "OUTDIR=!INDIR!!DATETAG!"
if not exist "!OUTDIR!" mkdir "!OUTDIR!"

echo.
echo  Input : %INFILE%
echo  Output: !OUTDIR!\!DATETAG!_NN!EXT!
echo.

for /f "delims=" %%D in ('ffprobe -v error -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "%INFILE%" 2^>nul') do set "DUR=%%D"
if defined DUR echo  Total length: ~%DUR% seconds
echo.

echo  Enter SPLIT points (where one patient ends / next begins).
echo  Format HH:MM:SS (e.g. 00:12:30), in order. Blank line to finish.
echo.

set /a COUNT=0
:GETSPLITS
set /a NEXT=COUNT+1
set "TS="
set /p "TS=  Split point #!NEXT! (blank to finish): "
if "!TS!"=="" goto SPLITSDONE
set "SPLIT[!COUNT!]=!TS!"
set /a COUNT+=1
goto GETSPLITS

:SPLITSDONE
echo.

set "PREV=00:00:00"
set /a PARTNUM=0

set /a I=0
:CUTLOOP
if %I% geq %COUNT% goto LASTPART
set /a PARTNUM+=1
call :PADNUM !PARTNUM!
set "CUR=!SPLIT[%I%]!"
set "OUT=!OUTDIR!\!DATETAG!_!PAD!!EXT!"
echo  [!DATETAG!_!PAD!]  !PREV!  -^>  !CUR!
ffmpeg -hide_banner -loglevel error -y -i "%INFILE%" -ss !PREV! -to !CUR! -c copy "!OUT!"
if errorlevel 1 echo     [WARN] ffmpeg issue on !DATETAG!_!PAD!.
set "PREV=!CUR!"
set /a I+=1
goto CUTLOOP

:LASTPART
set /a PARTNUM+=1
call :PADNUM !PARTNUM!
set "OUT=!OUTDIR!\!DATETAG!_!PAD!!EXT!"
echo  [!DATETAG!_!PAD!]  !PREV!  -^>  END
ffmpeg -hide_banner -loglevel error -y -i "%INFILE%" -ss !PREV! -c copy "!OUT!"
if errorlevel 1 echo     [WARN] ffmpeg issue on final part.

echo.
echo  ===============================================
echo   DONE. Created %PARTNUM% file(s) in:
echo   !OUTDIR!
echo  ===============================================
echo.
pause
endlocal
exit /b 0

:PADNUM
set "PAD=0%1"
set "PAD=!PAD:~-2!"
exit /b 0
