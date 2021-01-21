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
import pandas as pd
from clx_streamz_tools import utils
from clx_streamz_tools import streamz_workflow


class CybertWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(self, messages):
        # Messages will be received and run through cyBERT inferencing
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
        parsed_df, confidence_df = worker.data["cybert"].inference(df["stream"])
        confidence_df = confidence_df.add_suffix("_confidence")
        parsed_df = pd.concat([parsed_df, confidence_df], axis=1)
        return (parsed_df, batch_start_time, result_size)

    def worker_init(self):
        # Initialization for each dask worker
        from clx.analytics.cybert import Cybert

        worker = dask.distributed.get_worker()
        cy = Cybert()
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with cybert model. Model File: "
            + str(self.args.model)
            + " Label Map: "
            + str(self.args.label_map)
        )
        cy.load_model(self.args.model, self.args.label_map)
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"cybert": cy}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    cybert = CybertWorkflow()
    cybert.start()
