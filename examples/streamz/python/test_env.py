# Simple python script for testing that the Docker container has the appropriate software installed
from streamz import Stream
from confluent_kafka import Consumer, KafkaException
print("ENVIRONMENT OK")