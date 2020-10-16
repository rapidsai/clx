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
    echo "Example-1: bash $0 -w cybert"
    echo "Example-2: bash $0 -w dga"
    echo
    echo "Workflow env(kafka & dask) setup script"
    echo
    echo Positional:
    echo "  -w, --workflow      Envirnoment setup for a given workflow"
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
    -w|--workflow) shift; workflow=$1 ;;
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

	# Verify if workflow implementation exists
	has_workflow=false

	for path in ${CLX_STREAMZ_HOME}/python/*; do
	    if [[ -f $path ]]
		then
    		filename="$(basename "${path}")"
		filename=$(echo $filename | cut -f1 -d '.')
		if [ $2 = $filename ] ; then
       		   has_workflow=true
    		fi
	    fi
	done
	if [ ${has_workflow} = false ] ; then
    	   echo "Workflow '$workflow' doesn't exists"
    	   exit 1
	fi
  	log "INFO" "$1 = $2"
}

verify_input_arg "workflow" $workflow

# Intialize input variables
broker="localhost:9092"
zookeeper="localhost:2181"
input_topic="${workflow}_input"
output_topic="${workflow}_output"
input_data="${CLX_STREAMZ_HOME}/data/${workflow}_input.*"


#**********************************
# Configure Kafka
#**********************************
sed -i "/listeners=PLAINTEXT:\/\//c\listeners=PLAINTEXT:\/\/$broker" $KAFKA_HOME/config/server.properties
sed -i "/advertised.listeners=PLAINTEXT:\/\//c\advertised.listeners=PLAINTEXT:\/\/$broker" $KAFKA_HOME/config/server.properties
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
# Delete Kafka Topics if Exists
#**********************************
$KAFKA_HOME/bin/kafka-topics.sh --zookeeper $zookeeper --topic $input_topic --delete
$KAFKA_HOME/bin/kafka-topics.sh --zookeeper $zookeeper --topic $output_topic --delete

#**********************************
# Create Kafka Topics
#**********************************
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $input_topic
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $output_topic
log "INFO" "Kafka topics created"
#**********************************
# Read Sample Data
#**********************************
$KAFKA_HOME/bin/kafka-console-producer.sh --broker-list $broker --topic $input_topic < $input_data
log "INFO" "Sample ${workflow} data published to $input_topic kafka topic"

#**********************************
# Get available cuda devices
#**********************************
cuda_visible_devices_arr=( $(nvidia-smi| cut -d ' ' -f4| awk '$1 ~ /^[0-9]$/') )
cuda_visible_devices=$(IFS=,; echo "${cuda_visible_devices_arr[*]}")
log "INFO" "cuda_visible_devices list $cuda_visible_devices"

#**********************************
# Start Dask Scheduler
#**********************************
#log "INFO" "Starting Dask Scheduler at localhost:8787"
DASK_DASHBOARD_PORT=8787
DASK_SCHEDULER_PORT=8786
CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" nohup dask-scheduler --dashboard-address ${DASK_DASHBOARD_PORT} 2>&1 &
DASK_SCHEDULER_HOST=`hostname --ip-address`
DASK_SCHEDULER="${DASK_SCHEDULER_HOST}:${DASK_SCHEDULER_PORT}"
log "INFO" "Dask scheduler running"

#**********************************
# Start Dask CUDA Worker
#**********************************
for i in "${cuda_visible_devices_arr[@]}"
do
   echo "CUDA_VISIBLE_DEVICES=$i nohup dask-cuda-worker localhost:8786 2>&1 &"
   CUDA_VISIBLE_DEVICES=$i nohup dask-cuda-worker localhost:8786 2>&1 &
done
sleep 3