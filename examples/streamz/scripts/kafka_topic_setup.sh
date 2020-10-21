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
    echo "Example-1: bash $0 -b localhost:9092 -i cybert_input -o output_topic -d /data/path/to/publish/input_topic"
    echo "Example-2: bash $0 -i cybert_input -o output_topic -d /data/path/to/publish/input_topic"
    echo "Example-2: bash $0 -i cybert_input -o output_topic"
    echo
    echo "This script configures the kafka topic, such as creating and loading data."
    echo
    echo Positional:
    echo "  -b,  --broker           Kafka broker"
    echo "  -i,  --input_topic	    Input kafka topic"
    echo "  -o,  --output_topic		Output kafka topic"
    echo "  -d,  --data_path		Sample input data file path"
   	echo
    echo "  -h, --help		        Print this help"
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

# Verify user input arguments.
while [ $# != 0 ]; do
    case $1 in
    -h|--help) usage ;;
    -b|--broker) shift; input_topic=$1 ;;
    -i|--input_topic) shift; input_topic=$1 ;;
    -o|--output_topic) shift; output_topic=$1 ;;
    -d|--data_path) shift; data_path=$1 ;;
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

verify_input_arg "input_topic" $input_topic
verify_input_arg "output_topic" $output_topic 

# set default broker if not provided.
if [ -z "$broker" ]; then
	broker="localhost:9092"
	log "INFO" "Using default broker 'localhost:9092'" 
fi


#**********************************
# Create Kafka Topics
#**********************************
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $input_topic
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $broker --replication-factor 1 --partitions 1 --topic $output_topic
log "INFO" "Created '$input_topic' and '$output_topic kafka topics"

#**********************************
# Load Sample Data
#**********************************
if [ -f "$data_path" ]; then
	$KAFKA_HOME/bin/kafka-console-producer.sh --broker-list $broker --topic $input_topic < $data_path
	echo ""
	log "INFO" "Sample data at location '$data_path' is published to '$input_topic kafka' topic"
fi
