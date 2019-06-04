import logging

from factory.kafka_factory import KafkaFactory
from factory.nfs_factory import NFSFactory


class Factory:
    class InstanceGenerator(object):
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            class_name, config = self.func(*args, **kwargs)
            try:
                target_cls = globals()[class_name](config)
                return target_cls
            except:
                logging.error("unknown class name: { %s }" % (class_name))
                raise IOError("unknown class name: { %s }" % (class_name))

    @InstanceGenerator
    def getInstance(io_comp, config):
        io_comp = io_comp.lower()
        if io_comp == "kafka":
            return "KafkaFactory", config
        elif io_comp == "nfs":
            return "NFSFactory", config
        else:
            pass

    @staticmethod
    def getIOReader(io_comp, config):
        return Factory.getInstance(io_comp, config).getReader()

    @staticmethod
    def getIOWriter(io_comp, config):
        return Factory.getInstance(io_comp, config).getWriter()
