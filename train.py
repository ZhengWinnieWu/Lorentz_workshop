## Train the model
from random import sample as random_s
# train the model
import tensorflow as tf    
#tf.compat.v1.disable_v2_behavior() # <-- HERE !

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer, InputSpec

import json
import pickle

# cross-validation
def get_train_test_val(X_all, y_all, all_year):
    #test_year = [1988,1998,2006,2018,2020]     # select data only for testing and final model evaluation
    NY_train = 25                              # number of years for training
    #test_X = X_all.sel(time = X_all.time.dt.year.isin([test_year]))   # if test year has not been defined in the main module
    #test_y = y_all.sel(time = y_all.time.dt.year.isin([test_year]))
    #all_year = np.arange(SY,EY+1)
    #remain_year = set(all_year) - set(test_year)
    #train_year = random_s(remain_year, NY_train)
    train_year = random_s(all_year, NY_train)
    train_X = X_all.sel(time = X_all.time.dt.year.isin([train_year]))
    train_y = y_all.sel(time = y_all.time.dt.year.isin([train_year]))
    #val_year = set(remain_year) - set(train_year)
    val_year = set(all_year) - set(train_year)
    val_X = X_all.sel(time = X_all.time.dt.year.isin([list(val_year)]))
    val_y = y_all.sel(time = y_all.time.dt.year.isin([list(val_year)]))
    return train_X, train_y, val_X, val_y  #, test_X, test_y

def train_model(model, train_X, train_y, val_X, val_y, callbacks_path, epochs, batch_size, history_path, class_weight):
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=callbacks_path,
            monitor='val_loss',   # tf.keras.metrics.AUC(from_logits=True)
            save_best_only=True,
        )
    ]
    history = model.fit(
        train_X, train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True,
        shuffle=True,
        validation_data=(val_X, val_y),
        callbacks=callbacks_list,
        class_weight=class_weight
    )
    history = history.history

    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    
    return history

# Load History
def load_history(history_path):
    history = pickle.load(open(history_path, "rb"))
    return history
