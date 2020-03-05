# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set +e
#**************************
# Print usage instructions.
#**************************
usage() {
    local exitcode=0
    if [ $# != 0 ]; then
        echo "$@"
        exitcode=1
    fi
    echo "Usage: $0 [POS]... [ARG]..."
    echo
    echo "Example-1: bash $0 -p 8998 -w 2 -t 60"
    echo "Example-2: bash $0 --port 8990 --workers 4 --timeout 10"
    echo
    echo "CLX Query Service Runner..."
    echo
    echo Positional:
    echo "  -p, --port          Port number"
    echo "  -w, --workers       Number of workers"
    echo "  -t, --timeout       Application time out"
    echo
    echo "  -h, --help          Print this help"
    echo
    exit $exitcode
}

#*****************************
# This function print logging.
#*****************************
log(){
  if [[ $# = 2 ]]; then
     echo "$(date) [$1] : $2"
  fi
}

#Verify user input arguments.
while [ $# != 0 ]; do
    case $1 in
    -h|--help) usage ;;
    -p|--port) shift; port=$1 ;;
    -w|--workers) shift; workers=$1 ;;
    -t|--timeout) shift; timeout=$1 ;;
    -) usage "Unknown positional: $1" ;;
    -?*) usage "Unknown positional: $1" ;;
    esac
    shift
done

#**********************************
#Input arguments type & empty check
#**********************************
verify_input_arg(){ 
	if [[ -z $2 ]]; then
	  log "ERROR" "Argument '$1' is not provided" 
	  usage
	  exit 1
	fi
	
	if ! [[ "$2" =~ ^[0-9]+$ ]]
	    then
	        log "ERROR" "Argument '$1' accepts only numbers"
	        usage
	  		exit 1
	fi
}

verify_input_arg "port" $port
verify_input_arg "workers" $workers
verify_input_arg "timeout", $timeout

log "INFO" "Input arguments check passed!"

# Django application variables.
NAME="clx_query_service"
DJANGO_WSGI_MODULE=clx_query_service.wsgi
DJANGO_SETTINGS_MODULE=clx_query_service.settings   
BLZ_READER_CONF_PATH=conf/clx_blz_reader_conf.yaml                            
DJANGODIR=/rapids/clx/siem_integrations/clx_query_service

cd $DJANGODIR

export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DJANGODIR:$PYTHONPATH
export BLZ_READER_CONF=$DJANGODIR/$BLZ_READER_CONF_PATH

LOGS_PATH=$DJANGODIR/logs
LOG_FILE=query_service.log

# Verify logs directory.
if [[ ! -e $LOGS_PATH ]]; then
	log "INFO" "Directory $LOGS_PATH doesn't exists"
	log "INFO" "Creating directory $LOGS_PATH" 
    mkdir -p $LOGS_PATH
fi

log "INFO" "Starting '$NAME' with $workers workers running on port $port"

# Start django web application using gunicorn.
exec gunicorn ${DJANGO_WSGI_MODULE}:application \
     --name $NAME \
     --bind 0.0.0.0:$port \
     --workers $workers \
     --timeout $timeout \
     --log-level=INFO \
     --log-file=$LOGS_PATH/$LOG_FILE

set -e