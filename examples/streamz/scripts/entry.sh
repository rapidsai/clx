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
    echo "Example-1: bash $0 -b localhost:9092 -g streamz -i input -o output -m /path/to/model.pth -l /path/to/labels.yaml"
    echo
    echo "Run cybert model using kafka"
    echo
    echo Positional:
    echo "  -b, --broker               Kafka Broker"
    echo "  -g, --group_id             Kafka group ID"
    echo "  -i, --input_topic          Kafka input topic"
    echo "  -o, --output_topic         Kafka output topic"
    echo "  -m, --model_file           Cybert model file"
    echo "  -l, --label_file           Cybert label file"
    echo "  -c, --cuda_visible_devices Cuda visible devices, ex: 0,1,2"
    echo "  -d, --data                 Cybert data file"
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
    -g|--group_id) shift; group_id=$1 ;;
    -i|--input_topic) shift; input_topic=$1 ;;
    -o|--output_topic) shift; output_topic=$1 ;;
    -m|--model_file) shift; model_file=$1 ;;
    -l|--label_file) shift; label_file=$1 ;;
    -d|--data) shift; data=$1 ;;
    -c|--cuda_visible_devices) shift; data=$1 ;;
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
  log "INFO" "$1 = $2"
}

verify_input_arg "broker" $broker
verify_input_arg "group_id" $group_id
verify_input_arg "input_topic", $input_topic
verify_input_arg "output_topic", $output_topic
verify_input_arg "model_file", $model_file
verify_input_arg "label_file", $label_file

source activate clx

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
# Create Kafka Topics
#**********************************
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $input_topic
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $output_topic
log "INFO" "Kafka topics created"
#**********************************
# Read Sample Data
#**********************************
$KAFKA_HOME/bin/kafka-console-producer.sh --broker-list $broker --topic $input_topic < $data
log "INFO" "Sample data read into kafka topic, $input_topic"

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
nohup dask-cuda-worker localhost:8786 2>&1 &

#**********************************
# Start Jupyter Notebook
#**********************************
nohup jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root 2>&1 &

#**********************************
# Run Cybert
#**********************************
log "INFO" "Preparing to run cybert"
log "INFO" "python -i /python/cybert.py --input_topic input --output_topic output --group_id $group_id --model $model_file --label_map $label_file --dask_scheduler ${DASK_SCHEDULER}"
python -i /python/cybert.py --input_topic input --output_topic output --group_id $group_id --model $model_file --label_map $label_file --dask_scheduler ${DASK_SCHEDULER}