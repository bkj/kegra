from __future__ import print_function

import numpy as np
np.random.seed(123)

import keras.backend as K
from keras.layers import Input, Dropout, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution, GraphInput
from kegra.utils import *

import time

# --
# Params

DATASET = 'cora'
FILTER = 'localpool'
SYM_NORM = True
NB_EPOCH = 200
PATIENCE = 10

# --
# IO

X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
X = np.diag(1./np.array(X.sum(1)).flatten()).dot(X)
A_ = preprocess_adj(A, SYM_NORM)

# --
# Define model

def MergeDot(output_dim):
  return Lambda(lambda x: K.dot(x[1], x[0]), output_shape=(output_dim,))

feats_in = Input(shape=(X.shape[1],), name='feats_in')
graph_in = GraphInput(sparse=True, name='graph_in')

feats_drop = Dropout(0.5)(feats_in)

# First GCN layer
fg_dot = MergeDot(X.shape[1])([feats_drop, graph_in])
gc1 = Dense(16, activation='relu', bias=False)(fg_dot)
gc1 = Dropout(0.5)(gc1)

# Second GCN layer
fg_dot2 = MergeDot(16)([gc1, graph_in])
gc2 = Dense(y.shape[1], activation='softmax', bias=False)(fg_dot2)

model = Model(input=[feats_in, graph_in], output=gc2)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# --
# Train model

wait = 0
preds = None
best_val_loss = 99999

for epoch in range(NB_EPOCH):
    
    # Single training iteration (we mask nodes without labels for loss calculation)
    _ = model.fit(
      {"feats_in" : X, "graph_in" : A_}, y_train, 
      sample_weight = train_mask,
      batch_size = A.shape[0], 
      nb_epoch = 1, 
      shuffle = False, 
      verbose = 0
    )
    
    # Predict on full dataset
    preds = model.predict({"feats_in" : X, "graph_in" : A_}, batch_size=A.shape[0])
    
    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
    
    print(
      "Epoch: {:04d}".format(epoch),
      "train_loss= {:.4f}".format(train_val_loss[0]),
      "train_acc= {:.4f}".format(train_val_acc[0]),
      "val_loss= {:.4f}".format(train_val_loss[1]),
      "val_acc= {:.4f}".format(train_val_acc[1])
    )
    
    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
