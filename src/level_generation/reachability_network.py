import sys
import os
import numpy as np
import random
import argparse
from PIL import Image
from architectures import ReachabilityNetwork8LayersDeep as ReachabilityNetwork
import tensorflow as tf
from dataset import Dataset


# Experiment parameters
epoch_number = 120
learning_rate = 0.00005
batch_size = 128
image_height = 14
image_width = 28
channels = 1
epsilon = 0.0000001
examples_per_epoch = 5

def save_image_from_level(path, level):
    assert len(level.shape) == 2
    img = Image.fromarray((level * 255).astype('uint8'), mode='L')
    img.save(path)


def kl_divergence(distrib_a, distrib_b):
    assert distrib_a.shape.as_list() == distrib_b.shape.as_list() and distrib_a.dtype == distrib_b.dtype, \
        "Distributions should have the same shape: {} vs {} and the same dtype: {} vs {}".format(
            distrib_a.shape,
            distrib_b.shape,
            distrib_a.dtype,
            distrib_b.dtype
        )
    distrib_a = tf.maximum(distrib_a, tf.constant(epsilon))
    distrib_b = tf.maximum(distrib_b, tf.constant(epsilon))
    X = tf.distributions.Categorical(probs=distrib_a)
    Y = tf.distributions.Categorical(probs=distrib_b)
    return tf.reduce_mean(tf.distributions.kl_divergence(X, Y))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", type=str, required=True,
                        help="Path to the input samples")
    parser.add_argument("-l", "--labels", type=str, required=True,
                        help="Path to the input labels")
    parser.add_argument("-t", "--tensorboard", type=str, required=False,
                        help="Path to the directory to store tensorboard results")
    parser.add_argument("-e", "--examples", type=str, required=False,
                        help="Path to the directory to store some examples")
    parser.add_argument("-m", "--save_model", type=str, required=False,
                        help="Use if you want to save the model")
    parser.add_argument("-f", "--fraction", type=float, default=0.9,
                        help="TensorFlow per-process GPU memory fraction")
    parser.add_argument("--test", action="store_true",
                        help="Do not train, test datasets with the given model")

    args = parser.parse_args()

    # get path of samples
    samples_folder = args.samples
    print("Using samples from folder %s" % samples_folder)

    # get path of labels
    labels_folder = args.labels
    print("Using labels from folder %s" % labels_folder)

    # get path for tensorboard
    tb_folder = args.tensorboard
    if tb_folder:
        print("Saving results to tensorboard folder %s" % tb_folder)
    else:
        print("Not using tensorboard")
    # create tb folder if missing
    if tb_folder and not os.path.exists(tb_folder):
        os.makedirs(tb_folder)
        print("Tensorboard log directory created at %s" % tb_folder, "directory")

    # get path for examples
    examples_folder = args.examples
    if examples_folder:
        print("Saving examples to examples folder %s" % examples_folder)
    else:
        print("Not saving examples")
    # clean and re-create examples folder if necessary
    if examples_folder:
        if os.path.exists(examples_folder):
            import shutil
            shutil.rmtree(examples_folder)
        os.makedirs(examples_folder)
        print("Examples folder created at %s" % examples_folder)

    # get path for final model
    model_file = args.save_model

    # are we only doing a test?
    doing_only_test = args.test

    # create the network
    print("Defining the network")
    network = ReachabilityNetwork(trainable=not doing_only_test)

    # create the main placeholders
    print("Creating placeholders")
    input_tensor_placeholder = tf.placeholder(shape=(None, image_height, image_width, channels), dtype=tf.float32)
    label_tensor_placeholder = tf.placeholder(shape=(None, image_height, image_width, channels), dtype=tf.float32)
    curr_epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32)
    increment_curr_epoch_op = tf.assign_add(curr_epoch_var, 1)

    graph_nodes = dict()
    graph_nodes['G_probab_solid'] = input_tensor_placeholder

    # compute the output from the placeholders
    print("Creating network")
    logits = network(**graph_nodes)
    
    # probability of [non-reachable, reachable]
    label_tensor = tf.concat([1-label_tensor_placeholder, label_tensor_placeholder], axis=-1)

    output_probabilities = tf.nn.softmax(logits)
    # sample and discretize
    classes = tf.contrib.distributions.Categorical(logits=logits).sample()
    label_tensor_enc = tf.contrib.distributions.Categorical(probs=label_tensor).sample()

    # compute the loss
    print("Creating losses")
    discrete_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label_tensor_enc,
        logits=logits
    )
    continue_loss = tf.math.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_tensor,
            logits=logits
        )
    )
    loss = continue_loss

    # create session config object
    print("Defining session properties")
    tf_sess_config = tf.ConfigProto()
    tf_sess_config.gpu_options.allow_growth = True
    tf_sess_config.gpu_options.per_process_gpu_memory_fraction = args.fraction

    print("Defining metrics and metrics summaries")
    with tf.variable_scope('metrics', reuse=True):

        accuracy, accurancy_op = tf.metrics.accuracy(labels=label_tensor_enc, predictions=classes)
        precision, precision_op = tf.metrics.precision(labels=label_tensor_enc, predictions=classes)
        recall, recall_op = tf.metrics.recall(labels=label_tensor_enc, predictions=classes)

        metrics_summaries = tf.summary.merge([
            tf.summary.scalar(
                'accuracy',
                accuracy
            ),
            tf.summary.scalar(
                'precision',
                precision
            ),
            tf.summary.scalar(
                'recall',
                recall
            ),
            tf.summary.scalar(
                'f1',
                2 * recall * precision / (recall + precision)
            )
        ])
        kl_div = kl_divergence(label_tensor, tf.nn.softmax(logits, axis=-1))
        reachable_unreachable_ratio = tf.reduce_mean(tf.nn.softmax(logits, axis=-1))
        mean_square = tf.losses.mean_squared_error(label_tensor, tf.nn.softmax(logits, axis=-1))
        metrics_update_ops = [accurancy_op, precision_op, recall_op]
        metrics_ops = [accuracy, precision, recall]

    # Define initializer to initialize/reset running variables
    print("Getting local variables that should be resetted after each test session")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    print("Loading the datasets")
    if doing_only_test:
        data = Dataset(samples_folder, labels_folder, train_perc=0.0)
    else:
        data = Dataset(samples_folder, labels_folder, train_perc=0.7)

    if not doing_only_test:
        # get all variables that has to be trained
        print("Getting trainable variables")
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "")
        print("Trainable variables", theta)

        # optimizer
        print("Creating optimizer")
        adam_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=theta)
        gdo_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, var_list=theta)
        solver = adam_solver

        # define training operations
        training_ops = [loss, solver]

        print("Starting training")
        with tf.Session(config=tf_sess_config) as session:
            print("Initilizing variables")
            session.run(tf.global_variables_initializer())

            # define where to collect summaries
            if tb_folder:
                tb_writer = tf.summary.FileWriter(tb_folder, session.graph)

            # getting current epoch
            curr_epoch = session.run(curr_epoch_var)

            # training
            while curr_epoch < epoch_number:
                """
                # start training phase
                """
                losses = []
                data.init_training()
                while data.has_next_training():
                    samples, labels = data.next_training(batch_size)

                    feed_dict = {
                        input_tensor_placeholder: np.expand_dims(samples, axis=3),
                        label_tensor_placeholder: np.expand_dims(labels, axis=3)
                    }
                    
                    batch_loss, _ = session.run(training_ops, feed_dict=feed_dict)
                    losses.append(batch_loss)
                
                training_loss_results = np.mean(losses)

                if tb_folder:
                    summary_loss = tf.Summary()
                    summary_loss.value.add(tag="losses/training_loss", simple_value=training_loss_results)
                    tb_writer.add_summary(summary_loss, curr_epoch)


                """
                # start test phase
                """
                session.run(tf.local_variables_initializer())
                data.init_test()

                losses = []
                mean_squares = []
                kl_div_metrics = []
                ratios = []
                
                while data.has_next_test():
                    samples, labels = data.next_test(batch_size)

                    feed_dict = {
                        input_tensor_placeholder: np.expand_dims(samples, axis=3),
                        label_tensor_placeholder: np.expand_dims(labels, axis=3)
                    }
                    
                    batch_loss, batch_kl_div_metric, batch_reachable_unreachable_ratio, batch_mean_square = \
                        session.run([loss, kl_div, reachable_unreachable_ratio, mean_square], feed_dict=feed_dict)

                    session.run(metrics_update_ops, feed_dict=feed_dict)
                    # adding measures to lists
                    losses.append(batch_loss)
                    kl_div_metrics.append(batch_kl_div_metric)
                    ratios.append(batch_reachable_unreachable_ratio)
                    mean_squares.append(batch_mean_square)

                test_loss_results = np.mean(losses)

                if tb_folder:
                    summary_metric = tf.Summary()
                    summary_metric.value.add(tag="metrics/test_kl", simple_value=np.mean(kl_div_metrics))
                    tb_writer.add_summary(summary_metric, curr_epoch)

                    summary_metric = tf.Summary()
                    summary_metric.value.add(tag="metrics/test_reach_ratio", simple_value=np.mean(ratios))
                    tb_writer.add_summary(summary_metric, curr_epoch)

                    summary_metric = tf.Summary()
                    summary_metric.value.add(tag="metrics/test_mean_squared_error", simple_value=np.mean(mean_squares))
                    tb_writer.add_summary(summary_metric, curr_epoch)

                    summary_loss = tf.Summary()
                    summary_loss.value.add(tag="losses/test_loss", simple_value=test_loss_results)
                    tb_writer.add_summary(summary_loss, curr_epoch)

                    metrics_summaries_res = session.run(metrics_summaries)
                    tb_writer.add_summary(metrics_summaries_res, curr_epoch)

                '''
                generate some examples if requested
                ''' 
                data.init_test()
                samples, labels = data.next_test(examples_per_epoch)

                if examples_folder:
                    feed_dict = {
                        input_tensor_placeholder: np.expand_dims(samples, axis=3),
                        label_tensor_placeholder: np.expand_dims(labels, axis=3)
                    }
                    a, b = session.run([classes, label_tensor_enc], feed_dict=feed_dict)
                    for idx, val in enumerate(zip(a, b)):
                        save_image_from_level(os.path.join(examples_folder, 'sample_distrib-%d-%d.png' % (curr_epoch, idx)), np.squeeze(val[0]))
                        save_image_from_level(os.path.join(examples_folder, 'labels_distrib-%d-%d.png' % (curr_epoch, idx)), np.squeeze(val[1]))

                # update epoch number
                curr_epoch = session.run(increment_curr_epoch_op)
                print("Epoch: ", curr_epoch, ", training loss:", training_loss_results, ", test loss:", test_loss_results)

            if model_file:
                if os.path.exists(model_file):
                    os.remove(model_file)
                print("Saving trained model to %s" % model_file)
                network._save_weights(model_file)
                print("Model saved, training is ended, exiting")
            else:
                print("Training is done, model is not going to be saved, exiting")

    else:
        print("Starting testing")
        with tf.Session(config=tf_sess_config) as session:
            print("Initilizing variables")
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            network._restore_weights(model_file)

            data.init_test()

            losses = []
            mean_squares = []
            kl_div_metrics = []
            ratios = []

            while data.has_next_test():
                samples, labels = data.next_test(batch_size)

                feed_dict = {
                    input_tensor_placeholder: np.expand_dims(samples, axis=3),
                    label_tensor_placeholder: np.expand_dims(labels, axis=3)
                }

                batch_loss, batch_kl_div_metric, batch_reachable_unreachable_ratio, batch_mean_square = \
                    session.run([loss, kl_div, reachable_unreachable_ratio, mean_square], feed_dict=feed_dict)

                session.run(metrics_update_ops, feed_dict=feed_dict)
                # adding measures to lists
                losses.append(batch_loss)
                kl_div_metrics.append(batch_kl_div_metric)
                ratios.append(batch_reachable_unreachable_ratio)
                mean_squares.append(batch_mean_square)

                if examples_folder:
                    feed_dict = {
                        input_tensor_placeholder: np.expand_dims(samples, axis=3),
                        label_tensor_placeholder: np.expand_dims(labels, axis=3)
                    }
                    a, b = session.run([label_tensor, output_probabilities], feed_dict=feed_dict)
                    for idx, val in enumerate(zip(a, b)):
                        np.save(os.path.join(examples_folder, 'label_distrib-%d.npy' % idx), np.squeeze(val[0]))
                        np.save(os.path.join(examples_folder, 'out_distrib-%d.npy' % idx), np.squeeze(val[1]))

            test_loss_results = np.mean(losses)
            kl_div_results = np.mean(kl_div_metrics)
            ratios_results = np.mean(ratios)
            square_results = np.mean(mean_squares)
            accu, prec, rec = session.run(metrics_ops)

            print("- Results -")
            print("Loss mean: %4f" % test_loss_results)
            print("KL divergence: %4f" % kl_div_results)
            print("Reachable/nonReachable ratio: %4f" % ratios_results)
            print("Mean square error: %4f" % square_results)
            print("Accuracy: %4f" % accu)
            print("Precision: %4f" % prec)
            print("Recall: %4f" % rec)
            print("F1 score: %4f" % (2 * rec * prec / (rec + prec)))
            print("Test complete, exiting")
            

        





                

    
