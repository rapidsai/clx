# Copyright (c) 2019, NVIDIA CORPORATION.
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

import logging

from clx.io.factory.kafka_factory import KafkaFactory
from clx.io.factory.fs_factory import FileSystemFactory
from clx.io.factory.dask_fs_factory import DaskFileSystemFactory

log = logging.getLogger(__name__)


class Factory:

    __cls_dict = {
        "kafka": "KafkaFactory",
        "fs": "FileSystemFactory",
        "dask_fs": "DaskFileSystemFactory",
    }

    @staticmethod
    def cls_dict():
        return Factory.__cls_dict

    class InstanceGenerator(object):
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            class_name, config = self.func(*args, **kwargs)
            try:
                target_cls = globals()[class_name](config)
                return target_cls
            except KeyError as error:
                log.error(error)
                log.exception(error)
                raise

    @InstanceGenerator
    def get_instance(io_comp, config):
        io_comp = io_comp.lower()
        if io_comp and io_comp in Factory.cls_dict():
            return Factory.cls_dict()[io_comp], config
        else:
            raise KeyError(
                "Dictionary doesn't have { %s } corresponding component class."
                % (io_comp)
            )

    @staticmethod
    def get_reader(io_comp, config):
        return Factory.get_instance(io_comp, config).get_reader()

    @staticmethod
    def get_writer(io_comp, config):
        return Factory.get_instance(io_comp, config).get_writer()
