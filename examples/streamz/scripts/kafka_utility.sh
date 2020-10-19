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
    echo "Example-1: bash $0 -w cybert -dt true -ld true"
    echo "Example-3: bash $0 -w cybert -ld true"
    echo "Example-2: bash $0 -w dga -dt true"
    echo "Example-4: bash $0 -w dga"
    echo
    echo "This script provides workflow specific kafka topics and loads data to input topic"
    echo
    echo Positional:
    echo "  -w,  --workflow	        Prefix to create workflow specifc kafka topics"
    echo "  -dt, --delete_topic		Delete existing topics (Optional)"
    echo "  -ld, --load_data		Load data to input topic (Optional)"
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
    -w|--workflow) shift; workflow=$1 ;;
    -dt|--delete_topic) shift; delete_topic=$1 ;;
    -ld|--load_data) shift; load_data=$1 ;;
    -) usage "Unknown positional: $1" ;;
    -?*) usage "Unknown positional: $1" ;;
    esac
    shift
done

#*****************************************
# Verify workflow implementation 
#*****************************************
verify_workflow(){
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

verify_workflow "workflow" $workflow

# Intialize input variables
input_topic="${workflow}_input"
output_topic="${workflow}_output"

BROKER="localhost:9092"
ZOOKEEPER="localhost:2181"

#**********************************
# Delete Kafka Topics if Exists
#**********************************
if [ "$delete_topic" = true ] ; then
	$KAFKA_HOME/bin/kafka-topics.sh --zookeeper $ZOOKEEPER --topic $input_topic --delete
	$KAFKA_HOME/bin/kafka-topics.sh --zookeeper $ZOOKEEPER --topic $output_topic --delete
fi

#**********************************
# Create Kafka Topics
#**********************************
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $BROKER --replication-factor 1 --partitions 1 --topic $input_topic
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server $BROKER --replication-factor 1 --partitions 1 --topic $output_topic
log "INFO" "Created '$input_topic' and '$output_topic kafka topics"

#**********************************
# Load Sample Data
#**********************************
if [ "$load_data" = true ] ; then
	input_data="${CLX_STREAMZ_HOME}/data/${workflow}_input.*"
	$KAFKA_HOME/bin/kafka-console-producer.sh --broker-list $BROKER --topic $input_topic < $input_data
	echo ""
	log "INFO" "Sample ${workflow} data published to '$input_topic kafka' topic"
fi
