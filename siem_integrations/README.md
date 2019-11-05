# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;**CLX Siem Integration**</div>

The [RAPIDS](https://rapids.ai) CLX Siem Integration project provides the following features for Splunk:
 * Splunk2Kafka - Export data from Splunk to Kafka.
 * CLX Query

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/splunk2kafka/blob/master/README.md) ensure you are on the `master` branch.

## Splunk2Kafka

Splunk2Kafka is a plugin which can be installed into your Splunk instance. Once installed, the command `export2kafka` can be used within the Splunk query window as shown below.

### Quick Start

Use this Splunk query template to send data to your kafka instance.
```
index="my-index" | export2kafka topic=my-topic broker=10.0.0.0:9092
```

Additional query configuration options are detailed [here](https://github.com/rapidsai/clx-siem-integration/blob/master/splunk2kafka/export2kafka/README.md).
### Install splunk2kafka

Please install the following applications into your Splunk instance by following the instructions linked below. 
In order to utilize splunk2kafka, a running [Kafka](https://kafka.apache.org/) instance will also be required.

1. Install splunk_wrapper [README.md](https://github.com/rapidsai/clx-siem-integration/blob/master/splunk2kafka/splunk_wrapper/README.md)
2. Install export2kafka [README.md](https://github.com/rapidsai/clx-siem-integration/blob/master/splunk2kafka/export2kafka/README.md)


## CLX Query

TODO: Add documentation on clx query


## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/clx-siem-integration/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
