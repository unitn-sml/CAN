"""
This is the starting point to run the code, here you provide
the experiment file entirely describes the experiment, optionally
you can decide on the fraction of gpu memory to use, the DEBUG level
and if to use the old trainers instead of the new ones.
"""
import argparse
import os
import sys
import tensorboard
import tensorboard.backend.application as tb_conf
import tensorflow as tf

from utils.utils_common import init_environment
from experiment import Experiment
from utils import utils_tf, memory_saving_gradients


def main(args):
    experiment = Experiment()
    experiment.read(args.input, args.debug)
    logger = experiment["LOGGER"]
    logger.info("Running experiment: {}".format(args.input))

    logger.info("Using memory saving gradients: %s" % args.memgrad)
    if args.memgrad:
        # monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
        tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

    # setup tensorflow GPU options
    tf_sess_config = tf.ConfigProto()
    tf_sess_config.gpu_options.allow_growth = True
    tf_sess_config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    utils_tf.fix_tf_gpu_memory_allocation(tf_sess_config)

    # logging environment info
    logger.info("TensorFlow path: {}".format(tf.__file__))
    logger.info("TensorFlow version: {}".format(tf.__version__))
    logger.info("Available GPUs: {}\n".format(str(utils_tf.get_available_gpus())))
    logger.debug("TensorFlow ConfigProto configuration:")
    for a in utils_tf.get_config_proto_attributes(tf_sess_config):
        logger.debug(a)

    # tensorboard related stuff
    try:
        logger.info("TensorBoard path: {}".format(tensorboard.__file__))
        logger.info("TensorBoard conf path: {}".format(tb_conf.__file__))
    except AttributeError:  # elder tensorboard packages lack those attributes
        pass

    tb_default_conf = tb_conf.DEFAULT_TENSOR_SIZE_GUIDANCE
    logger.info("TensorBoard conf: {}".format(tb_default_conf))
    tb_params = [tb_conf.scalar_metadata.PLUGIN_NAME,
                 tb_conf.image_metadata.PLUGIN_NAME,
                 tb_conf.histogram_metadata.PLUGIN_NAME]
    msg = "TensorBoard parameter '{}' is {}. Set it to 0 to see all summaries."
    for param in filter(lambda p: tb_default_conf[p] != 0, tb_params):
        logger.warn(msg.format(param, tb_default_conf[param]))

    # init seeds and eventually create folders
    init_environment(experiment)

    # run the experiment
    if args.evaluation_mode is None:
        # training mode
        experiment.run(tf_sess_config, args.complete_logging)
    else:
        # run in evaluation mode, generate only samples without training (requires an already trained model in the default folder)
        experiment.evaluation(tf_sess_config, args.complete_logging, args.evaluation_mode)

    logger.info("Done!")


if __name__ == '__main__':
    # so that we will be always working in the src directory (helps within the HPC cluster for stuff like
    # getting data etc. ~ you can run main from wherever you want, stuff should work
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the experiment.json file")
    parser.add_argument("-f", "--fraction", type=float, default=0.9,
                        help="TensorFlow per-process GPU memory fraction")
    parser.add_argument("-d", "--debug", type=bool, required=False,
                        default=False, help="Enables TensorBoard debugger")
    parser.add_argument("--memgrad", help="If memory saving gradients are to be used.", action="store_true",
                        required=False, default=False)
    parser.add_argument("--complete_logging", help="If seed logging should continue during training (costly).",
                        action="store_true", required=False, default=False)
    parser.add_argument("--evaluation_mode", type=int, required=False,
                        help="Run in evaluation mode, use this parameter to specity an integer number of samples that should be generated")

    args = parser.parse_args()
    main(args)
