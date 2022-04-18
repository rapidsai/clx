# <div align="left"><img src="../img/rapids_logo.png" width="90px"/>&nbsp;CLX SIEM Integration</div>

[RAPIDS](https://rapids.ai) CLX [SIEM](https://en.wikipedia.org/wiki/Security_information_and_event_management) Integrations provide features that enable interoperability between SIEMs and a RAPIDS/CLX environment. Currently, this support includes `splunk2kafka`, enabling data integration between Splunk and CLX.

## Splunk2Kafka

Splunk2Kafka is a plugin which can be installed into your Splunk instance. Once installed, the command `export2kafka` can be used within the Splunk query window as shown below.

### Quick Start

Use this Splunk query template to send data to your Kafka instance.
```
index="my-index" | export2kafka topic=my-topic broker=10.0.0.0:9092
```

Additional query configuration options are detailed [here](splunk2kafka/export2kafka/README.md).

### Install Splunk2Kafka

Install the following applications into your Splunk instance by following the instructions linked below. 
In order to utilize splunk2kafka, a [running Kafka instance](https://kafka.apache.org/quickstart) is required.

1. Install splunk_wrapper ([Instructions](splunk2kafka/splunk_wrapper/README.md))
2. Install export2kafka ([Instructions](splunk2kafka/export2kafka/README.md))


## CLX Query

### Overview
CLX Query is a [Splunk](https:/www.splunk.com) application that requests CLX Query Service ([Django](https:/www.djangoproject.com) RESTful service) to execute queries using CLX python package and get results back to the Splunk GUI. CLX python package uses [BlazingSQL](https://blazingsql.com) engine internally to execute queries. [Gunicorn](https:/gunicorn.org) provides Web Server Gateway Interface to forward requests to CLX Query Service. [Supervisor](http:/supervisor.org) monitors Gunicorn Web Server to ensure it doesn't go offline.

### Requirements
- django
- gunicorn
- supervisor
- djangorestframework

Above packages can be installed using either Conda or source code. Conda installation is preferred as it uses binaries. 
- Conda Install
    ```aidl
    conda install -c conda-forge django=3.0.3 gunicorn=20.0.4 supervisor=4.1.0 djangorestframework=3.11.0
    ```
- Install from the source
    ```aidl
    pip install django==3.0.3 gunicorn==20.0.4 supervisor==4.1.0 djangorestframework==3.11.0
    ```
### Download sample dataset
Download MovieLens stable benchmark [dataset](https://grouplens.org/datasets/movielens/25m/). It has 25 million ratings and one million tag applications applied to 62,000 movies by 162,000 users. Includes tag genome data with 15 million relevance scores across 1,129 tags. 

### Install and Run CLX Query Service
 
1. Update property `ALLOWED HOSTS` in `clx_query_service/clx_query_service/settings.py` with ip address of machine where CLX Query Service is planned to run. Example if docker container with CLX Query Service is running on host `5.56.114.13` then property will be like this `ALLOWED_HOSTS=["5.56.114.13"]`.

2. As we have downloaded sample MovieLens dataset. Now update the configuration file `clx_query_service/conf/clx_blz_reader_conf.yaml` with the location of the dataset. Provide suitable table name for the dataset which will be used in the queries. The `header` property is added as workaround for issue [blazingsql-265](https://github.com/BlazingDB/blazingsql/issues/265).
3. CLX Query Service Runner usage.

    ```aidl
    bash clx_query_service/bin/start_service.sh --help
    ```
    ``` 
    Usage: clx_query_service/bin/start_service.sh [POS]... [ARG]...
    Example-1: bash clx_query_service/bin/start_service.sh -p 8998 -w 2 -t 60
    Example-2: bash clx_query_service/bin/start_service.sh --port 8990 --workers 4 --timeout 10
    
    CLX Query Service Runner
    
    Positional:
      -p, --port          Port number
      -w, --workers       Number of workers
      -t, --timeout       Application time out
    
      -h, --help          Print this help
    ```
5. Run Gunicorn wrapped CLX Query Service application as daemon process using Supervisor.
   1. Update `command` property in the `clx_query_service/conf/clx_query_service.conf` as per the user requirements. Such as service binding `port`, number of `workers` and request `timeout` seconds.
   2. Copy update configuration file to supervisor location.
        ```aidl
        cp clx_query_service/conf/clx_query_service.conf /etc/supervisor
        ```
   3. Add configuration file to Supervisord.
        ```aidl
        supervisord -c /etc/supervisor/clx_query_service.conf
        ```
   4.  Add configuration file to Supervisorctl.
        ```aidl
        supervisorctl -c /etc/supervisor/clx_query_service.conf
        ```
   5.  Start CLX Query Service from Supervisorctl CLI.
        ```aidl
        start clx_query_service
        ```
### Install and Run CLX Query

1. To establish communication between CLX Query and CLX Query Service, update the configuration file `clx query/default/clx_query_setup.conf` with the CLX Query Service hostname and the port number. The value of the `hostname` property should be the same as the IP used in the porperty `ALLOWED HOSTS` of `clx_query_service/clx_query_service/settings.py`. The value of the `port` property should be the same as the number used in the `command` property of `clx_query_service/conf/clx_query_service.conf`.

2. Copy CLX Query to splunk apps directory.
    ```aidl
    cp -R clx_query splunk/etc/apps
    ```
3. Copy `splunklib` from [splunk-sdk-python](https://github.com/splunk/splunk-sdk-python) to splunk apps directory. Use tag version that matches your Splunk installation. *Note: Application was tested with Splunk 1.6.x*.
4. Restart splunk application server to take effect on changes.
    ```aidl
    ./splunk/bin/splunk restart
    ``` 
5. Login to Splunk GUI and launch CLX Query application. `Apps> Manage Apps> ClX Query> Launch App`
6. Run sample query
    -  Get number of user_id's and their average rating in descending order for each genre and title. Consider movies only with rating greater than 2.5.
        ```
        | clx query="SELECT genres, title, avg(rating) as avg_rating, count(user_id) as user_cnt from (SELECT main.movies.title as title, main.movies.genres as genres, main.ratings.userId as user_id, main.ratings.rating as rating FROM main.movies INNER JOIN main.ratings ON (main.ratings.movieId = main.movies.movieId) WHERE main.ratings.rating > 2.5) as tmp GROUP BY genres, title ORDER BY user_cnt DESC, avg_rating DESC"
        ```
      
        ![clx_query_screeshot](/siem_integrations/clx_query/clx_query.png)

### Known Issues
1.  Columns not being inferred from CSV header [blazingsql-265](https://github.com/BlazingDB/blazingsql/issues/265).

## Contributing Guide

Review the [CONTRIBUTING.md](../CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
