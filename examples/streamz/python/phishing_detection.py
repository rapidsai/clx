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

import time
import dask
import cudf
from clx_streamz_tools import utils
from clx_streamz_tools import streamz_workflow


class PhisingDetectionWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(messages):
        # Messages will be received and run through sequence classifier inferencing
        worker = dask.distributed.get_worker()
        batch_start_time = int(round(time.time()))
        df = cudf.DataFrame()
        if type(messages) == str:
            df["stream"] = [messages.decode("utf-8")]
        elif type(messages) == list and len(messages) > 0:
            df["stream"] = [msg.decode("utf-8") for msg in messages]
        else:
            print("ERROR: Unknown type encountered in inference")

        result_size = df.shape[0]
        print("Processing batch size: " + str(result_size))
        pred, prob = worker.data["seq_classifier"].predict(df["stream"])
        results_gdf = cudf.DataFrame({"pred": pred, "prob": prob})
        return (results_gdf, batch_start_time, result_size)

    def worker_init():
        # Initialization for each dask worker
        from clx.analytics.sequence_classifier import SequenceClassifier

        worker = dask.distributed.get_worker()
        seq_classifier = SequenceClassifier()
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with sequence classifier model. Model File: "
            + str(self.args.model)
        )
        seq_classifier.init_model(self.args.model)
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"seq_classifier": seq_classifier}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    phishing_detection = PhisingDetectionWorkflow()
    phishing_detection.start()
