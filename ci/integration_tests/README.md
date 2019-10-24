# CLX Integration Testing

CLX integrates with [Kafka](https://kafka.apache.org/) for the ability to read and write data from/to a Kafka queue. An integration test environment has been created to simulate and test this interaction.

## Running the Integration Test

To run the integration test simply run the following. This will run the integration test `run_integration_test.py`.

```
cd ci/integration_tests
docker-compose -f docker-compose.test.yml up
```

To continue re-running the integration tests, don't forget to first destroy your current docker images/containers, before creating a new one.

```
cd ci/integration_tests
docker-compose down
```