# Copyright (c) 2020, NVIDIA CORPORATION.
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

#*****************************
# This function print logging.
#*****************************
log(){
  if [[ $# = 2 ]]; then
     echo "$(date) [$1] : $2"
  fi
}

source activate rapids

# kakfa broker
BROKER="localhost:9092"

#**********************************
# Configure Kafka
#**********************************
sed -i "/listeners=PLAINTEXT:\/\//c\listeners=PLAINTEXT:\/\/$BROKER" $KAFKA_HOME/config/server.properties
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
log "INFO" "Kafka broker is running on $BROKER"
log "INFO" "Zookeeper is running on localhost:2181"

#**********************************
# Create topics and publish data
#**********************************
log "INFO" "Loading cybert input data to 'cybert_input' topic"
. $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
       -i cybert_input \
       -o cybert_output \
       -d $CLX_STREAMZ_HOME/data/apache_raw_sample_1k.txt

log "INFO" "Loading dga detection input data to 'dga_detection_input' topic"
. $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
       -i dga_detection_input \
       -o dga_detection_output \
       -d $CLX_STREAMZ_HOME/data/dga_detection_input.jsonlines
       
#**********************************
# Get available cuda devices
#**********************************
cuda_visible_devices_arr=( $(nvidia-smi | sed 1d| cut -d ' ' -f 4| awk '$1 ~ /^[0-9]$/') )
cuda_visible_devices=$(IFS=,; echo "${cuda_visible_devices_arr[*]}")
log "INFO" "cuda_visible_devices list $cuda_visible_devices"

#**********************************
# Start Dask Scheduler
#**********************************
log "INFO" "Starting Dask Scheduler at localhost:8786"
log "INFO" "Starting Dask Dashboard at localhost:8787"
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

exec "$@";
