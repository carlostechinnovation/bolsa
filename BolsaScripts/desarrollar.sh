#!/bin/bash

################ PROGRAMAS DE DESARROLLO (solo Linux) #############################################
LOG_DESA="/bolsa/logs/desarrollar.log"
DIR_PROGRAMAS="/home/carloslinux/Desktop/PROGRAMAS/"

echo -e "Abriendo Eclipse..." >> ${LOG_DESA}
${DIR_PROGRAMAS}eclipse/eclipse &

echo -e "Abriendo Pycharm..." >> ${LOG_DESA}
${DIR_PROGRAMAS}pycharm/bin/pycharm.sh &

echo -e "Abriendo SmartGIT..." >> ${LOG_DESA}
${DIR_PROGRAMAS}smartgit/bin/smartgit.sh &

echo -e "Abriendo DROPBOX..." >> ${LOG_DESA}
${DIR_PROGRAMAS}dropbox/.dropbox-dist/dropboxd &

