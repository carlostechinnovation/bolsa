#!/bin/bash

################ PROGRAMAS DE DESARROLLO (solo Linux) #############################################
LOG_DESA="/bolsa/logs/desarrollar.log"
DIR_PROGRAMAS="/home/carloslinux/Desktop/PROGRAMAS/"

echo -e "Abriendo Eclipse..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}eclipse/eclipse &
eclipse &

echo -e "Abriendo Pycharm..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}pycharm/bin/pycharm.sh &
pycharm-community &

echo -e "Abriendo SmartGIT..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}smartgit/bin/smartgit.sh &

#echo -e "Abriendo DROPBOX..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}dropbox/.dropbox-dist/dropboxd &
#~/.dropbox-dist/dropboxd &

echo -e "Abriendo terminal Linux..." >> ${LOG_DESA}
cd /bolsa/logs/
gnome-terminal &
cd /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/
gnome-terminal &

echo -e "Abriendo GitExtensions..." >> ${LOG_DESA}
${DIR_PROGRAMAS}GitExtensions/GitExtensions.exe &

#echo -e "Abriendo SQLiteStudio..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}SQLiteStudio/sqlitestudio &

