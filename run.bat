@echo off
echo ============================================
echo  Lancement du Systeme
echo ============================================
echo.

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Menu
:menu
echo.
echo Que voulez-vous faire?
echo 1. Telecharger le dataset
echo 2. Lancer l'entrainement complet
echo 3. Tester l'inference
echo 4. Lancer Federated Learning SEULEMENT
echo 5. Quitter
echo.
set /p choice="Votre choix (1-5): "

if "%choice%"=="1" goto download
if "%choice%"=="2" goto train
if "%choice%"=="3" goto inference
if "%choice%"=="4" goto train_fl
if "%choice%"=="5" goto end
goto menu

:download
echo.
echo Telechargement du dataset...
python download_data.py
pause
goto menu

:train
echo.
echo Lancement de l'entrainement...
python main.py
pause
goto menu

:inference
echo.
echo Lancement de l'inference...
python inference.py
pause
goto menu

:train_fl
echo.
echo Lancement de Federated Learning SEULEMENT...
python train_fl_only.py
pause
goto menu

:end
echo.
echo Au revoir!
exit