from __future__ import print_function

import json
import sys
import time

import confluent_kafka
from confluent_kafka import Producer

# import pprint
from splunklib.searchcommands import Configuration, Option, StreamingCommand, dispatch


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@Configuration(local=True)
class FileSinkCommand(StreamingCommand):
    broker = Option(require=True)
    topic = Option(require=True)
    batch = Option(require=False, default=2000)
    timeout = Option(require=False, default=60)
    pool = Option(require=False, default=2)
    start_time = int(time.time())

    def create_producers(self, pool, broker):
        producers = []
        for i in range(pool):
            producers.append(
                Producer({"bootstrap.servers": broker, "session.timeout.ms": 10000})
            )
            eprint("exprot2kafka - producer" + str(i) + " created: " + broker)
        return producers

    def stream(self, records):
        topic = str(self.topic)
        broker = str(self.broker)
        batch = int(self.batch)
        timeout = int(self.timeout)
        pool = int(self.pool)
        eprint(
            "export2kafka - starting... broker("
            + broker
            + ") topic("
            + topic
            + ") batch("
            + str(batch)
            + ") timeout("
            + str(timeout)
            + " mins) pool("
            + str(pool)
            + ")"
        )
        eprint("export2kafka - stream starting")
        producers = self.create_producers(pool, broker)
        cnt = 0

        for record in records:
            trimmed = {k: v for k, v in record.iteritems()}
            # eprint(json.dumps(trimmed))
            producers[cnt % pool].produce(topic, json.dumps(trimmed))
            cnt += 1

            if cnt % batch == 0:
                # batch level reached poll to get producer to move messages out
                eprint(
                    "export2kafka - batch reached, calling poll... processed records: "
                    + str(cnt)
                )
                for p in producers:
                    p.poll(0)

            if cnt % 10 == 0 and int(time.time()) > (60 * timeout) + self.start_time:
                # quit after timeout has been reached, only check every 10 records
                eprint("export2kafka - timeout reached, stopping search...")
                break

            # return record for display in Splunk
            yield record

        eprint(
            "export2kafka - all records processed for stream... processed records: "
            + str(cnt)
        )
        eprint("export2kafka - calling flush...")
        for p in producers:
            p.flush()
        eprint("export2kafka - flush finished...")
        eprint("export2kafka - stream finished")


if __name__ == "__main__":
    dispatch(FileSinkCommand, sys.argv, sys.stdin, sys.stdout, __name__)
