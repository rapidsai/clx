# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
from distributed import Client
from dask_cuda import LocalCUDACluster


def create_dask_client(dask_scheduler):
    # If a dask scheduler is provided create client using that address
    # otherwise create a new dask cluster
    if dask_scheduler is not None:
        print("Dask scheduler:", dask_scheduler)
        client = Client(dask_scheduler)
    else:
        cluster = LocalCUDACluster()
        client = Client(cluster)
    print(client)
    return client

def signal_term_handler(signal, frame):
    # Receives signal and calculates benchmark if indicated in argument
    print("Exiting streamz script...")
    if args.benchmark:
        (time_diff, throughput_mbps, avg_batch_size) = calc_benchmark(
            output, args.benchmark
        )
        print("*** BENCHMARK ***")
        print(
            "Job duration: {:.3f} secs, Throughput(mb/sec):{:.3f}, Avg. Batch size(mb):{:.3f}".format(
                time_diff, throughput_mbps, avg_batch_size
            )
        )
    sys.exit(0)
    
def calc_benchmark(processed_data, size_per_log):
    # Calculates benchmark for the streamz workflow
    t1 = int(round(time.time() * 1000))
    t2 = 0
    size = 0.0
    batch_count = 0
    # Find min and max time while keeping track of batch count and size
    for result in processed_data:
        (ts1, ts2, result_size) = (result[0], result[1], result[2])
        if ts1 == 0 or ts2 == 0:
            continue
        batch_count = batch_count + 1
        t1 = min(t1, ts1)
        t2 = max(t2, ts2)
        size += result_size * size_per_log
    time_diff = t2 - t1
    throughput_mbps = size / (1024.0 * time_diff) if time_diff > 0 else 0
    avg_batch_size = size / (1024.0 * batch_count) if batch_count > 0 else 0
    return (time_diff, throughput_mbps, avg_batch_size)