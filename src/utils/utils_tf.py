import os
import tensorflow as tf

from inspect import getmembers, isbuiltin, ismethod
from tensorflow.python.client import device_lib


def escape_snapshot_name(name):
    # square brackets cause a (de)serialization issue for TF checkpoints
    return name.replace("[", "(").replace("]", ")")


def fix_tf_gpu_memory_allocation(tf_sess_config):
    # TensorFlow 1.4.0 bug prevents setting GPU memory options for sessions
    # after listing available devices (device_lib.list_local_devices()).
    # A workaround is to create and destroy a temporary well-configured session
    # before listing available devices
    # ref: https://github.com/tensorflow/tensorflow/issues/9374
    # ref: https://github.com/tensorflow/tensorflow/issues/8021

    # unfortunately, this fix may cause another bug on windows (the issue
    # occurs when multiple sessions with a specific
    # per_process_gpu_memory_fraction are started
    if os.name != "nt":
        with tf.Session(config=tf_sess_config):  # bug-fix
            pass


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def get_config_proto_attributes(config_proto):
    # return public attributes of default ConfigProto object
    return [i for i in getmembers(config_proto)
            if not i[0].startswith('_') and not ismethod(i[1])
            and not isbuiltin(i[1])]


def pad_left(tensor):
    return tf.expand_dims(tensor, axis=0)
    
