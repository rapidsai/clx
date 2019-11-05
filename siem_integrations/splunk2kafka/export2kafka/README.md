# export2kafka

## Overview

Splunk App that installs `export2kafka` for use with alerts and searches to
export data from Splunk to Kafka

## Pre-reqs

1. Install Kakfa libraries
```
    sudo -i -u splunk bash
    source activate root
    conda install -c conda-forge python-confluent-kafka
    conda remove python-confluent-kafka
    conda install -c conda-forge librdkafka=0.11.0
    conda install -f -c conda-forge python-confluent-kafka
```
2. Setup `/etc/hosts` for the Kafka brokers
 
## Install

1. Git clone this repo into `$SPLUNKHOME/etc/apps`
2. Go to `http://$SPLUNKURL/en-us/debug/refresh`
3. Click **Refresh** button to load the app into the Web UI

## Usage
### Config Options
**broker**  
Usage - set a Kafka broker to use for bootstrap  
Required? - YES  
Format - <ip_addr>:<port>  
Example - broker=10.0.0.0:9092  
  
**topic**  
Usage - set the Kafka topic to publish to  
Required? - YES  
Format - <topic>  
Example - topic=data_raw
  
**batch**  
Usage - set the batch size before calling poll on producer  
Required? - NO  
Format - integer  
Default - 2000 records  
Example - batch=2000  
  
**timeout**  
Usage - set the timeout of the export in minutes  
Required? - NO  
Format - integer in minutes  
Default - 60 mins  
Example - timeout=60  
  
**pool**  
Usage - set the number of producers used, useful when exporting large data sets  
Required? - NO  
Format - integer  
Default - 2 producers  
Example - pool=2  

### Query Example

```
index="my-index" | export2kafka topic=my-topic broker=10.0.0.0:9092
```
