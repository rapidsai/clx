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
from clx_streamz_tools import utils
from clx_streamz_tools import streamz_workflow


class DGADetectionWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(self, messages_df):
        # Messages will be received and run through DGA inferencing
        worker = dask.distributed.get_worker()
        batch_start_time = int(round(time.time()))
        result_size = messages_df.shape[0]
        print("Processing batch size: " + str(result_size))
        dd = worker.data["dga_detector"]
        preds = dd.predict(messages_df["domain"])
        messages_df["preds"] = preds
        return (messages_df, batch_start_time, result_size)

    def worker_init(self):
        # Initialization for each dask worker
        from clx.analytics.dga_detector import DGADetector

        worker = dask.distributed.get_worker()
        dd = DGADetector()
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with dga model. Model File: "
            + str(self.args.model)
        )
        dd.load_model(self.args.model)
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"dga_detector": dd}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    dga_detection = DGADetectionWorkflow()
    dga_detection.start()
