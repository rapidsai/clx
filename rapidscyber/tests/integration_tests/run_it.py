import logging
import time
import multiprocessing
logging.basicConfig(filename="run_it.out", level=logging.DEBUG)

from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions, ConfigResource, ConfigSource
from confluent_kafka import Producer, Consumer
from rapidscyber.workflow import netflow_workflow

bootstrap_server = "kafka:29092"
topics = ["input", "cyber-enriched-data"]
msgs = ["cyber test 1", "cyber test 2"]
producer = None
consumer = None

def setup():
    print("Begin test setup...")
    create_kafka_topics()
    global producer
    producer = create_producer()
    wf = create_workflow()
    print("Test setup complete.")

def create_kafka_topics():
    # Create Kafka topics
    kafka_admin = AdminClient({'bootstrap.servers': bootstrap_server})
    kafka_topics = [NewTopic(topic, num_partitions=1, replication_factor=1) for topic in topics]
    fs = kafka_admin.create_topics(kafka_topics)
    print("Kafka topics created.")
    return fs

def create_producer():
    # Send cyber messages to input kafka topic
    producer_conf = {
        "bootstrap.servers": bootstrap_server,
        "session.timeout.ms": 10000,
    }
    producer = Producer(producer_conf)
    print("Kafka producer created.")
    return producer

def create_workflow(input_topics, output_topic, boostrap_server):
    # Create workflow
    source = {
        "type": "kafka",
        "kafka_brokers": boostrap_server,
        "group_id": "cyber-dp",
        "batch_size": 1,
        "consumer_kafka_topics": input_topics,
        "time_window": 5
    }
    dest = {
        "type": "kafka",
        "kafka_brokers": boostrap_server,
        "group_id": "cyber-dp",
        "batch_size": 1,
        "publisher_kafka_topic": output_topic,
        "output_delimiter": ","
    }
    wf = netflow_workflow.NetflowWorkflow(source=source, destination=dest, name="my-kafka-workflow")
    print("Workflow created.")
    return wf

def run_workflow():
    print("Running netflow workflow.")
    wf.run_workflow()

def send_data():
    for msg in msgs:
        print("Sending msg to workflow: " + msg)
        producer.produce(topic_names[0], msg)

def verify():
    print("teardown")

def teardown():
    print("Verifying messages were processed by workflow...")
    consumer_conf = {
        "bootstrap.servers": "kafka:29092",
        "group.id": "int-test",
        "session.timeout.ms": 10000,
        "default.topic.config": {"auto.offset.reset": "largest"}
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(topic_names[1])
    for msg in msgs:
        enriched_msg = consumer.poll(timeout=1.0)
        print("Enriched msg processed... " + enriched_msg)

def main():
    setup()
    process_run_workflow = multiprocessing.Process(target=run_workflow)
    process_send_data = multiprocessing.Process(target=send_data)
    process_run_workflow.start()
    process_send_data.start()
    time.sleep(10)
    process_send_data.terminate()
    process_run_workflow.terminate()
    verify()
    teardown()
