#!/bin/bash

NAME="clx_query_service"                             
DJANGODIR=/rapids/clx/siem_integrations/clx_query_service  
NUM_WORKERS=2  
DJANGO_SETTINGS_MODULE=clx_query_service.settings 
DJANGO_WSGI_MODULE=clx_query_service.wsgi  
BLZ_READER_CONF_PATH=conf/clx_blz_reader_conf.yaml 
LOG_FILE=query_service.log
TIMEOUT=60

cd $DJANGODIR
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DJANGODIR:$PYTHONPATH
export BLZ_READER_CONF=$DJANGODIR/$BLZ_READER_CONF_PATH

LOGS_PATH=$DJANGODIR/logs

if [[ ! -e $LOGS_PATH ]]; then
    mkdir -p $LOGS_PATH
fi

exec gunicorn ${DJANGO_WSGI_MODULE}:application \
     --name $NAME \
     --bind 0.0.0.0:8998 \
     --workers $NUM_WORKERS \
     --timeout $TIMEOUT \
     --log-level=DEBUG \
     --log-file=$LOGS_PATH/$LOG_FILE
