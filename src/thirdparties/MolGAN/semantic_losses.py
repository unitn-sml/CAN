import tensorflow as tf

import os
from py3psdd import Vtree, SddManager, PSddManager, io


class SemanticLoss:
    def __init__(self, use_sigmoid):
        self._use_sigmoid = use_sigmoid
        self.constraint_name = self.__class__.__name__

    @staticmethod
    def _import_psdd(constraint_name):
        constraint_name = constraint_name.replace("SemanticLoss_", "")
        vtree = "constraints/" + constraint_name + ".vtree"
        sdd = "constraints/" + constraint_name + ".sdd"
        assert os.path.isfile(vtree), vtree + " is not a file."
        assert os.path.isfile(sdd), sdd + " is not a file."

        # load vtree and sdd files and construct the PSDD
        vtree = Vtree.read(vtree)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd, manager)
        pmanager = PSddManager(vtree)
        psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)
        return psdd

    def forward(self, nodes, edges, z):
        print("SEMANTIC LOSS FORWARD")
        if self.use_sigmoid:
            nodes = tf.nn.sigmoid(nodes, name=self.constraint_name + "_to_probabilities")
            edges = tf.nn.sigmoid(edges, name=self.constraint_name + "_to_probabilities")
        bs = tf.shape(nodes)[0]

        tmpz = z[:, -5:]
        tmpz = tf.reshape(tmpz, [bs, 1, 5])
        znodestacked = tf.concat([tmpz, nodes], axis=1)
        znodestacked = tf.reshape(znodestacked, [bs, 10 * 5])

        print("importing inputToTypeAtom")
        inputToTypeAtompsdd = self.__class__._import_psdd("inputToTypeAtom")
        inputToTypeAtomwmc = inputToTypeAtompsdd.generate_tf_ac_v2(znodestacked)
        inputToTypeAtomwmc = tf.reshape(inputToTypeAtomwmc, [bs, 1])

        logging_dict = dict()

        input_to_atom_loss = self.to_wmc_loss(bs, [inputToTypeAtomwmc])
        logging_dict["wmc_input_to_atom"] = 1.0 - input_to_atom_loss
        loss = input_to_atom_loss
        return loss, logging_dict

    def to_wmc_loss(self, bs, wmcs_list):
        lastdim = sum([node.shape[1] for node in wmcs_list])
        wmc = tf.concat(wmcs_list, axis=1)
        wmc = tf.reshape(wmc, [bs, lastdim])
        wmc = tf.reduce_prod(wmc, axis=1)
        wmc = tf.reduce_mean(wmc)
        return 1.0 - wmc

    @property
    def use_sigmoid(self):
        """
        Is sigmoid being applied to G_output_logit?
        :return:
        """
        return self._use_sigmoid

    @use_sigmoid.setter
    def use_sigmoid(self, value):
        """
        Set if sigmoid should be applied to G_output_logit.
        :return:
        """
        assert (isinstance(value, bool))
        self._use_sigmoid = value
