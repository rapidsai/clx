#!/bin/bash
set +e

#**************************
# Print usage instructions.
#**************************

usage() {
    
    echo "Usage: $0 [POS]... [ARG]..."
    echo
    echo "Example-1: bash $0"
    echo "Example-2: bash $0 -b localhost:9092"
    echo
    echo "Kafka and Dask runner"
    echo
    echo Optional:
    echo "  -b, --broker        Kafka Broker"
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
    -b|--broker) shift; broker=$1 ;;
    -) usage "Unknown optional: $1" ;;
    -?*) usage "Unknown optional: $1" ;;
    esac
    shift
done

#**********************************
#Input arguments type & empty check
#**********************************
verify_input_arg(){
	if [[ -z $2 ]]; then
	  local broker="localhost:9092"
	  echo $broker
	else
          echo $2
        fi
}

broker=$(verify_input_arg "broker" $broker)
log "INFO" "Using kafka broker '$broker'"

source activate rapids

#**********************************
# Configure Kafka
#**********************************
sed -i "/#listeners=PLAINTEXT:\/\/:9092/c\listeners=PLAINTEXT:\/\/$broker" $KAFKA_HOME/config/server.properties
sed -i "/#advertised.listeners=PLAINTEXT:\/\/your.host.name:9092/c\advertised.listeners=PLAINTEXT:\/\/$broker" $KAFKA_HOME/config/server.properties
log "INFO" "Kafka configuration updated"
#**********************************
# Run Kafka and Zookeeper
#**********************************
$KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties
sleep 3
$KAFKA_HOME/bin/kafka-server-start.sh -daemon $KAFKA_HOME/config/server.properties
sleep 3
log "INFO" "Kafka and zookeeper running"

#**********************************
# Start Dask Scheduler
#**********************************
#log "INFO" "Starting Dask Scheduler at localhost:8787"
DASK_DASHBOARD_PORT=8787
DASK_SCHEDULER_PORT=8786
nohup dask-scheduler --dashboard-address ${DASK_DASHBOARD_PORT} 2>&1 &
DASK_SCHEDULER_HOST=`hostname --ip-address`
DASK_SCHEDULER="${DASK_SCHEDULER_HOST}:${DASK_SCHEDULER_PORT}"
log "INFO" "Dask scheduler running"

#**********************************
# Start Dask CUDA Worker
#**********************************
nohup dask-cuda-worker localhost:8786 2>&1 &

sleep 3

