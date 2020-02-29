# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;CLX SIEM Integration</div>

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/clx/blob/master/README.md) ensure you are on the `master` branch.

[RAPIDS](https://rapids.ai) CLX [SIEM](https://en.wikipedia.org/wiki/Security_information_and_event_management) Integrations provide features that enable interoperability between SIEMs and a RAPIDS/CLX environment. Currently, this support includes `splunk2kafka`, enabling data integration between Splunk and CLX.

## Splunk2Kafka

Splunk2Kafka is a plugin which can be installed into your Splunk instance. Once installed, the command `export2kafka` can be used within the Splunk query window as shown below.

### Quick Start

Use this Splunk query template to send data to your Kafka instance.
```
index="my-index" | export2kafka topic=my-topic broker=10.0.0.0:9092
```

Additional query configuration options are detailed [here](https://github.com/rapidsai/clx/blob/master/splunk2kafka/export2kafka/README.md).

### Install Splunk2Kafka

Install the following applications into your Splunk instance by following the instructions linked below. 
In order to utilize splunk2kafka, a [running Kafka instance](https://kafka.apache.org/quickstart) is required.

1. Install splunk_wrapper ([Instructions](https://github.com/rapidsai/clx/blob/master/splunk2kafka/splunk_wrapper/README.md))
2. Install export2kafka ([Instructions](https://github.com/rapidsai/clx-siem-integration/blob/master/splunk2kafka/export2kafka/README.md))


## CLX_Query

CLX Query is a splunk application with the ability to perform customized queries on the clx python module, which can trigger internal workflows to retrieve the requested analytical report using blazingsql engine.

### Quick Start

Use this splunk query template to generate report.
```
| clx query="<query goes here...>"
```

### Install CLX_Query

1. Update configuration file with hostname and port number of clx restful services.

    ```aidl
    vi clx_query/default/clx_query_setup.conf
    ```
2. Copy CLX_Query to splunk apps directory.

    ```aidl
    cp -R clx_query splunk/etc/apps
    ```
3. Restart splunk application server to take effect on changes.

    ```aidl
    ./splunk/bin/splunk restart
    ```

## CLX_Query_Service


### Install CLX_Query_Service


## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/clx/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
