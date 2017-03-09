from __future__ import print_function

from time import time
from scipy import sparse as sp
import numpy as np
np.random.seed(123)

import keras.backend as K
from keras import metrics
from keras.layers import Input, Dropout, Lambda, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# --
# Helpers

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(map(classes_dict.get, labels), dtype=np.int32)
    return labels_onehot

def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    idx_features_labels = np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.loadtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(map(idx_map.get, edges_unordered.flatten()), dtype=np.int32).reshape(edges_unordered.shape)
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    
    return features.todense(), adj, labels


def make_mask(idx, n):
    mask = np.zeros(n)
    mask[idx] = 1
    return mask == 1


def get_splits(y):
    idx_train = range(140)
    idx_val   = range(200, 500)
    idx_test  = range(500, 1500)
    
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_val[idx_val] = y[idx_val]
    
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test

# --
# IO

X, adj, y = load_data(dataset='cora')

y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y)
train_mask = make_mask(idx_train, y.shape[0])

# L1 Norm rows
X /= X.sum(axis=1)

adj = adj + sp.eye(adj.shape[0]) # Add loops
d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten()) # symmetric normalization
adj_norm = adj.dot(d).T.dot(d).tocsr()

# --
# Define model

def masked_metric(inds, metric):
  def f(y_true, y_pred):
    return metric(y_true[inds], y_pred[inds])
  return f

def make_metrics():
  return [
    masked_metric(idx_train, metrics.categorical_accuracy), 
    masked_metric(idx_train, metrics.categorical_crossentropy),
    
    masked_metric(idx_val, metrics.categorical_accuracy),
    masked_metric(idx_val, metrics.categorical_crossentropy),
    
    masked_metric(idx_test, metrics.categorical_accuracy),
    masked_metric(idx_test, metrics.categorical_crossentropy),
  ]

def MergeDot(output_dim):
  return Lambda(lambda x: K.dot(x[1], x[0]), output_shape=(output_dim,))

# Params
np.random.seed(456)
INPUT_DIM = X.shape[1]
N_NODES = adj.shape[1]
HIDDEN_DIM = 16

# Topology
feats_in = Input(shape=(INPUT_DIM,), name='feats_in')
graph_in = Input(shape=(N_NODES,), sparse=True, name='graph_in')

d1 = Dense(HIDDEN_DIM, activation='relu', bias=False)
d2 = Dense(y.shape[1], activation='softmax', bias=False)

gc1 = d1(feats_in)
gc2 = d2(MergeDot(HIDDEN_DIM)([gc1, graph_in]))

model = Model(input=[feats_in, graph_in], output=gc2)

model.layers[1].trainable = True
model.layers[-1].trainable = True

try:
  opt = model.optimizer
  K.set_value(opt.lr, K.get_value(opt.lr) / 10)
except:
  opt = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=make_metrics())

model.metrics_names = ['loss', 'train_acc', 'train_ent', 'val_acc', 'val_ent', 'test_acc', 'test_ent']

# --
# Train

NB_EPOCH = 200
PATIENCE = 10

Xm = adj_norm.dot(X)

_ = model.fit(
  {"feats_in" : Xm, "graph_in" : adj_norm}, y, 
  sample_weight = train_mask.astype('int'),
  batch_size = N_NODES, 
  nb_epoch = NB_EPOCH, 
  shuffle = False,
  verbose = 2,
  callbacks = [EarlyStopping(monitor='val_ent', patience=PATIENCE)]
)

# !! Need to add early stopping

print("\nFinal:")
res = model.evaluate({"feats_in" : Xm, "graph_in" : adj_norm}, y, verbose=False, batch_size=adj_norm.shape[0])
for k,v in zip(model.metrics_names, res):
  print('%.04f\t%s' % (v, k))


# --


