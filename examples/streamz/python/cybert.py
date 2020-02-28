
import cudf
import dask
import json
from dask_cuda import LocalCUDACluster
from distributed import Client
from streamz import Stream

# Define some global configurations
broker = 'localhost:9092'
topic = "input"
consumer_group = "streamz"
consumer_conf = {'bootstrap.servers': broker,
                 'group.id': consumer_group, 'session.timeout.ms': 60000}

# TODO: Takes the RAW Windows Event logs from Kafka and runs NER predictions against each message.
def predict_batch(messages):
    return messages

# TODO: Takes the prediction results and creates a cuDF from them.
def wel_parsing(predictions):
    return predictions

# TODO: Monitor for thresholds and trigger alert by letting those messages pass through here
def threshold_alert(event_logs):
    return event_logs

def worker_init():
    import clx

if __name__ == '__main__':
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    client.run(worker_init)
    # Define the streaming pipeline.
    source = Stream.from_kafka_batched(topic, consumer_conf, poll_interval='1s',
                                    npartitions=1, asynchronous=True, dask=False)
    inference = source.map(predict_batch)
    wel_parsing = inference.map(wel_parsing)
    alerts = wel_parsing.map(threshold_alert).sink(print)
    # Start the stream.
    source.start()
    



