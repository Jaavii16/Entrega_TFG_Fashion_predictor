@echo off
echo Activando entorno virtual...

REM Activar el entorno virtual
call "C:\Users\Usuario\Documents\Javier 2025-26\TFG\env11\Scripts\activate.bat"

echo Ejecutando script...
python "C:\Users\Usuario\Documents\Javier 2025-26\TFG\Codigo\Finales\recolector_tendencias_google.py"

echo Proceso completado.
