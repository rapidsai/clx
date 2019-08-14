import sys
import time
import logging
logging.basicConfig(filename="run_it.out", level=logging.DEBUG)
import threading
from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions, ConfigResource, ConfigSource
from confluent_kafka import Producer, Consumer
from rapidscyber.workflow import netflow_workflow

bootstrap_server = "kafka:29092"
topics = ["input", "cyber-enriched-data"]
msg = "cyber test"
consumer = None
wf = None
produce = None

def setup():
    print("Begin test setup...")
    create_kafka_topics()
    global wf
    wf = create_workflow([topics[0]], topics[1], bootstrap_server)
    print("Test setup complete.")

def create_kafka_topics():
    # Create Kafka topics
    kafka_admin = AdminClient({'bootstrap.servers': bootstrap_server})
    kafka_topics = [NewTopic(topic, num_partitions=1, replication_factor=1) for topic in topics]
    fs = kafka_admin.create_topics(kafka_topics)
    print("Kafka topics created.")
    return fs

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
    # Send cyber messages to input kafka topic
    producer_conf = {
        "bootstrap.servers": bootstrap_server,
        "session.timeout.ms": 10000,
    }
    producer = Producer(producer_conf)
    print("Kafka producer created.")
    for i in range(1, 13):
        time.sleep(1)
        message = '%s %s'%(msg, i) 
        print("Sending msg to workflow: " + message)
        producer.produce(topics[0], message)

def teardown():
    print("teardown")

def verify():
    print("Verifying messages were processed by workflow...")
    consumer_conf = {
        "bootstrap.servers": bootstrap_server,
        "group.id": "int-test",
        "session.timeout.ms": 10000,
        "default.topic.config": {"auto.offset.reset": "smallest"}
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([topics[1]])
    # Adding extra iteration would allow consumer to prepare and start polling messages.
    for i in range(1,25):
        enriched_msg = consumer.poll(timeout=1.0)
        if enriched_msg is not None and not enriched_msg.error():
            data = enriched_msg.value().decode("utf-8")
            print("Enriched msg processed... " + data)

def main():
    setup()
    t_run_workflow = threading.Thread(target=run_workflow, name='t_run_workflow') 
    t_send_data = threading.Thread(target=send_data, name='t_send_data')
    t_run_workflow.daemon = True
    t_run_workflow.start()
    time.sleep(15)
    t_send_data.start()
    t_send_data.join()
    verify()
    teardown()
    sys.exit()

if __name__ == "__main__":
    main()