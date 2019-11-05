import sys
import time
import logging
import threading
from confluent_kafka.admin import (
    AdminClient,
    NewTopic,
    NewPartitions,
    ConfigResource,
    ConfigSource,
)
from confluent_kafka import Producer, Consumer
from clx.workflow import netflow_workflow

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("integration_test")

kafka_bootstrap_server = "kafka:9092"
input_test_topics = ["input"]
output_test_topic = "cyber-enriched-data"
test_messages = ["cyber test {iter}".format(iter=i) for i in range(1, 13)]
test_workflow = None


def setup(input_topics, output_topic, bootstrap_server):
    """
    Setup required to begin integration test including creation of kafka topics and creation of a test workflow
    """
    log.info("Begin test setup..." + output_topic + " " + bootstrap_server)
    all_topics = ["input", "cyber-enriched-data"]
    create_kafka_topics(all_topics, bootstrap_server)
    global test_workflow
    test_workflow = create_workflow(input_topics, output_topic, bootstrap_server)
    log.info("Test setup complete.")


def create_kafka_topics(topics, bootstrap_server):
    """
    Creates the provided kafka topics given the kafka bootstrap server
    """
    log.info("Creating kafka topics... ")
    print(topics)
    # Create Kafka topics
    kafka_admin = AdminClient({"bootstrap.servers": bootstrap_server})
    kafka_topics = [
        NewTopic(topic, num_partitions=1, replication_factor=1) for topic in topics
    ]
    fs = kafka_admin.create_topics(kafka_topics)
    log.info("Kafka topics created... " + str(fs))


def create_workflow(input_topics, output_topic, bootstrap_server):
    """
    Creates the provided workflow given the input kafka topics, output topic, and provided bootstrap server
    """
    log.info("Creating test Netflow workflow...")
    source = {
        "type": "kafka",
        "kafka_brokers": bootstrap_server,
        "group_id": "cyber-dp",
        "batch_size": 1,
        "consumer_kafka_topics": input_topics,
        "time_window": 5,
    }
    dest = {
        "type": "kafka",
        "kafka_brokers": bootstrap_server,
        "group_id": "cyber-dp",
        "batch_size": 1,
        "publisher_kafka_topic": output_topic,
        "output_delimiter": ",",
    }
    workflow = netflow_workflow.NetflowWorkflow(
        source=source, destination=dest, name="my-kafka-workflow"
    )
    log.info("Workflow created... " + workflow.name)
    return workflow


def run_workflow(workflow):
    """
    Runs the given workflow
    """
    log.info("Running netflow workflow.")
    workflow.run_workflow()


def send_test_data(bootstrap_server, topic):
    """
    Sends test messages to the provided kafka bootstrap server and kafka topic
    """
    producer_conf = {"bootstrap.servers": bootstrap_server, "session.timeout.ms": 10000}
    producer = Producer(producer_conf)
    log.info("Kafka producer created.")
    for message in test_messages:
        time.sleep(1)
        log.info("Sending msg to workflow: " + message)
        producer.produce(topic, message)
    producer.flush()


def verify(bootstrap_server, output_topic):
    """
    Verifies that messages were processed by the workflow
    """
    log.info("Verifying messages were processed by workflow...")
    consumer_conf = {
        "bootstrap.servers": bootstrap_server,
        "group.id": "int-test",
        "session.timeout.ms": 10000,
        "default.topic.config": {"auto.offset.reset": "earliest"},
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([output_topic])
    # Adding extra iteration would allow consumer to prepare and start polling messages.
    expected_messages = set(
        ["{0},netflow_enriched".format(msg) for msg in test_messages]
    )
    start_time = time.time()
    while expected_messages:
        enriched_msg = consumer.poll(timeout=1.0)
        if enriched_msg is not None and not enriched_msg.error():
            data = enriched_msg.value().decode("utf-8")
            log.info("Enriched msg processed... " + data)
            if data in expected_messages:
                expected_messages.remove(data)
        if (time.time() - start_time) > 60:
            raise TimeoutError("Integration test did not complete.")


def main():
    global input_test_topics, output_test_topic, kafka_bootstrap_server
    print(input_test_topics)
    setup(input_test_topics, output_test_topic, kafka_bootstrap_server)
    # Start thread for running workflow
    global test_workflow
    t_run_workflow = threading.Thread(
        target=run_workflow, args=(test_workflow,), name="t_run_workflow"
    )
    t_run_workflow.daemon = True
    t_run_workflow.start()
    time.sleep(15)
    # Start thread for sending test data to kafka
    t_send_data = threading.Thread(
        target=send_test_data,
        args=(kafka_bootstrap_server, input_test_topics[0]),
        name="t_send_data",
    )
    t_send_data.start()
    t_send_data.join()
    # Verify that expected messages have been processed by the workflow
    verify(kafka_bootstrap_server, output_test_topic)


if __name__ == "__main__":
    main()
