#!/bin/bash

################ PROGRAMAS DE DESARROLLO (solo Linux) #############################################
DIR_BASE="/bolsa/"
LOG_DESA="${DIR_BASE}logs/desarrollar.log"
DIR_PROGRAMAS="/home/carloslinux/Desktop/PROGRAMAS/"

echo -e "Abriendo Eclipse..." >> ${LOG_DESA}
${DIR_PROGRAMAS}eclipse/eclipse &

echo -e "Abriendo Pycharm..." >> ${LOG_DESA}
${DIR_PROGRAMAS}pycharm/bin/pycharm.sh &

echo -e "Abriendo SmartGIT..." >> ${LOG_DESA}
${DIR_PROGRAMAS}smartgit/bin/smartgit.sh &

#echo -e "Abriendo DROPBOX..." >> ${LOG_DESA}
#${DIR_PROGRAMAS}dropbox/.dropbox-dist/dropboxd &
~/.dropbox-dist/dropboxd &

echo -e "Abriendo terminal Linux..." >> ${LOG_DESA}
gnome-terminal

echo -e "Abriendo GitExtensions..." >> ${LOG_DESA}
${DIR_PROGRAMAS}GitExtensions/GitExtensions.exe &
