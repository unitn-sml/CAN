import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer
import argparse
import os
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=32, required=False)
parser.add_argument("-d", "--dropout", type=float, default=0.0, required=False)
parser.add_argument("--n_critic", type=int, default=5, required=False)
parser.add_argument("--metrics", type=str, default="logp,sas,qed", required=False)
parser.add_argument("--n_samples", type=int, default=5000, required=False)
parser.add_argument("-z", "--z_dim", type=int, default=32, required=False)
parser.add_argument("-l", "--lambd", type=float, default=1.0, required=False)
parser.add_argument("--lambd_SL", type=float, default=1.0, required=False)
parser.add_argument("-e", "--epochs", type=int, default=300, required=False)
parser.add_argument("-a", "--activation_epoch", type=int, default=300, required=False)
parser.add_argument("--activation_epoch_SL", type=int, default=300, required=False)
parser.add_argument("-s", "--save_every", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--batch_discriminator", type=bool, default=True, required=False)
parser.add_argument("--name", type=str, default="./output", required=True)
parser.add_argument("--sl_use_sigmoid", type=bool, default=False, required=False)
parser.add_argument("--discrete_z", type=int, default=0, required=False)
parser = parser.parse_args()

batch_dim = parser.batch_size
dropout = parser.dropout
n_critic = parser.n_critic

"""
QED = druglikeness
logp = solubility
sas = synthetizability
"""
metric = parser.metrics
# metric = 'validity'
n_samples = parser.n_samples
z_dim = parser.z_dim

la = parser.lambd
la_SL = parser.lambd_SL
epochs = parser.epochs
past_epoch = parser.activation_epoch
past_epoch_SL = parser.activation_epoch_SL
save_every = parser.save_every
lr = parser.lr
batch_discriminator = parser.batch_discriminator
sl_use_sigmoid = parser.sl_use_sigmoid
name = parser.name

data = SparseMolecularDataset()
# data.load('data/gdb9_9nodes.sparsedataset')
data.load('data/qm9_5k.sparsedataset')

steps = (len(data) // batch_dim)

if not os.path.exists(name):
    os.makedirs(name)

with open("%s/parameters.txt" % name, "w") as file:
    for arg in vars(parser):
        print(arg, getattr(parser, arg))
        file.write("%s = %s\n" % (arg, getattr(parser, arg)))


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    a = [optimizer.train_step_G] if i % n_critic == 0 else [optimizer.train_step_D]
    b = [optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
    return a + b


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    embeddings = model.sample_z(batch_dim)

    if la < 1 or la_SL < 1:

        if i % n_critic == 0:
            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > past_epoch else 1.0,
                         optimizer.la_SL: la_SL if epoch > past_epoch_SL else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > past_epoch else 1.0,
                         optimizer.la_SL: la_SL if epoch > past_epoch_SL else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: dropout,
                     optimizer.la: 1.0,
                     optimizer.la_SL: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    dict = {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la, 'loss SL': optimizer.loss_SL}
    for name, wmc in optimizer.SL_log_dict.items():
        dict[name] = wmc
    return dict


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def reward(mols):
    rr = 1.
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

        if m == 'np':
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        elif m == 'novelty':
            rr *= MolecularMetrics.novel_scores(mols, data)
        elif m == 'dc':
            rr *= MolecularMetrics.drugcandidate_scores(mols, data)
        elif m == 'unique':
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= MolecularMetrics.diversity_scores(mols, data)
        elif m == 'validity':
            rr *= MolecularMetrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


# model
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((128, 64), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=True,
                      hard_gumbel_softmax=False,
                      batch_discriminator=batch_discriminator,
                      discrete_z=parser.discrete_z)

# optimizer
optimizer = GraphGANOptimizer(model, learning_rate=lr, feature_matching=False, sl_use_sigmoid=sl_use_sigmoid)

# session
session = tf.Session()
session.run(tf.global_variables_initializer())

# trainer
trainer = Trainer(model, optimizer, session)

print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

trainer.train(batch_dim=batch_dim,
              epochs=epochs,
              steps=steps,
              train_fetch_dict=train_fetch_dict,
              train_feed_dict=train_feed_dict,
              eval_fetch_dict=eval_fetch_dict,
              eval_feed_dict=eval_feed_dict,
              test_fetch_dict=test_fetch_dict,
              test_feed_dict=test_feed_dict,
              save_every=save_every,
              directory=name,
              _eval_update=_eval_update,
              _test_update=_test_update)
