import os
import random
import warnings

import cv2
import gdown
from functools import partial

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pandas as pd
from matplotlib import cm
from numpy.random import rand
import matplotlib.pyplot as plt
import jax
from jax import jit
from jax import random
from jax import numpy as jnp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
import pathlib

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    keras.mixed_precision.set_global_policy("mixed_float16")
except:
    pass

seed = 1337
tf.random.set_seed(seed)

import numpy as np
from keras.datasets import mnist
from keras.models import Model
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
from glob import glob
from sklearn.metrics import roc_curve, auc
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_auc_score
from tensorflow.keras import callbacks 
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,BatchNormalization,Dropout
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,GlobalAveragePooling2D,BatchNormalization,Dropout
%matplotlib inline
import shutil
from sklearn.metrics import  precision_score, recall_score, accuracy_score,classification_report ,confusion_matrix
from tensorflow.python.platform import build_info as tf_build_info
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


data_dir = pathlib.Path("./Data/TrainingOAMASK")
data_dir_ = pathlib.Path("./Data/ValOAJPG")

class Parameters:
    # data level
    image_size = 480
    batch_size = 4
    num_grad_accumulation = 8
    label_smooth=0.05
    class_number = 3
    val_split = 0.995
    train_split = 0.0005
    verbosity = 1
    autotune = tf.data.AUTOTUNE
    
    # hparams
    epochs = 125
    lr_sched = 'cosine_restart' # [or, exponential, cosine, linear, constant]
    lr_base  = 0.00076
    lr_min   = 0
    lr_decay_epoch  = 2.4
    lr_warmup_epoch = 3
    lr_decay_factor = 0.97
    
    scaled_lr = lr_base * (batch_size / 256.0)
    scaled_lr_min = lr_min * (batch_size / 256.0)
    num_validation_sample = int(image_count * val_split)
    num_training_sample = image_count - num_validation_sample
    train_step = int(np.ceil(num_training_sample / float(batch_size)))
    total_steps = train_step * epochs

params = Parameters()

train_set = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=train_split,
    subset="training",
    label_mode='categorical',
    shuffle=False,
    seed=params.image_size,
    image_size=(params.image_size, params.image_size),
    batch_size=params.batch_size,
)

val_set = keras.utils.image_dataset_from_directory(
    data_dir_,
    validation_split=val_split,
    subset="validation",
    label_mode='categorical',
    seed=params.image_size,
    image_size=(params.image_size, params.image_size),
    batch_size=params.batch_size,
)

tcls_names, vcls_names = train_set.class_names , val_set.class_names

tf_to_keras_augment = keras.Sequential(
    [
        RandomApply(layers.RandomFlip("horizontal"), probability=0.5),
    ],
    name="tf2keras_augment",
)

keras_aug = keras.Sequential(
    [
        layers.Resizing(height=params.image_size, width=params.image_size),
    ],
    name="keras_augment",
)

train_ds = train_set.shuffle(10 * params.batch_size)
train_ds = train_ds.map(
    lambda x, y: (keras_aug(x), y), num_parallel_calls=params.autotune
)

train_ds = train_ds.prefetch(buffer_size=params.autotune)
val_ds = val_set.prefetch(buffer_size=params.autotune)

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

class WarmupLearningRateSchedule(optimizers.schedules.LearningRateSchedule):
    """WarmupLearningRateSchedule a variety of learning rate
    decay schedules with warm up.
    
    Ref. https://gist.github.com/innat/69e8f3500c2418c69b150a0a651f31dc
    """

    def __init__(
        self,
        initial_lr,
        steps_per_epoch=None,
        lr_decay_type="exponential",
        decay_factor=0.97,
        decay_epochs=2.4,
        total_steps=None,
        warmup_epochs=3,
        minimal_lr=0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_lr = initial_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay_type = lr_decay_type
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.total_steps = total_steps
        self.warmup_epochs = warmup_epochs
        self.minimal_lr = minimal_lr

    def __call__(self, step):
        if self.lr_decay_type == "exponential":
            assert self.steps_per_epoch is not None
            decay_steps = self.steps_per_epoch * self.decay_epochs
            lr = schedules.ExponentialDecay(
                self.initial_lr, decay_steps, self.decay_factor, staircase=True
            )(step)
            
        elif self.lr_decay_type == "cosine":
            assert self.total_steps is not None
            lr = (
                0.5
                * self.initial_lr
                * (1 + tf.cos(np.pi * tf.cast(step, tf.float32) / self.total_steps))
            )

        elif self.lr_decay_type == "linear":
            assert self.total_steps is not None
            lr = (1.0 - tf.cast(step, tf.float32) / self.total_steps) * self.initial_lr

        elif self.lr_decay_type == "constant":
            lr = self.initial_lr

        elif self.lr_decay_type == "cosine_restart":
            decay_steps = self.steps_per_epoch * self.decay_epochs
            lr = tf.keras.experimental.CosineDecayRestarts(
                self.initial_lr, decay_steps
            )(step)
        else:
            assert False, "Unknown lr_decay_type : %s" % self.lr_decay_type

        if self.minimal_lr:
            lr = tf.math.maximum(lr, self.minimal_lr)

        if self.warmup_epochs:
            warmup_steps = int(self.warmup_epochs * self.steps_per_epoch)
            warmup_lr = (
                self.initial_lr
                * tf.cast(step, tf.float32)
                / tf.cast(warmup_steps, tf.float32)
            )
            lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: lr)

        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "lr_decay_type": self.lr_decay_type,
            "decay_factor": self.decay_factor,
            "decay_epochs": self.decay_epochs,
            "total_steps": self.total_steps,
            "warmup_epochs": self.warmup_epochs,
            "minimal_lr": self.minimal_lr,
        }
    
learning_rate_ = WarmupLearningRateSchedule(
    params.scaled_lr,
    steps_per_epoch=params.train_step,
    decay_epochs=params.lr_decay_epoch,
    warmup_epochs=params.lr_warmup_epoch,
    decay_factor=params.lr_decay_factor,
    lr_decay_type=params.lr_sched,
    total_steps=params.total_steps,
    minimal_lr=params.scaled_lr_min,
)

import tensorflow as tf
convnext = tf.keras.applications.convnext.ConvNeXtLarge(
    model_name='convnext_large',
    include_top=False,
    include_preprocessing=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(480, 480, 3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
conv = convnext.layers[-1].output
conv = Dropout(0.5)(conv)
output = Flatten()(conv)
output = Dense(3, activation='softmax')(output)
model_ = Model(inputs=convnext.input, outputs=output)

# 构建 checkpoint file

checkpoint_path = "./checkpoint/ConvNeXt/ConvNeXt.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True, 
                                      verbose=1, 
                                      monitor="val_accuracy",
                                      save_best_only=True,
                                      mode="max",
                                      )

log = callbacks.CSVLogger("history.csv", separator=",", append=False)


opt1=tf.keras.optimizers.Adam(learning_rate=learning_rate_, epsilon=0.1)

model_.compile(optimizer=opt1,
            loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])

history = model_.fit(
        train_ds,
        epochs=params.epochs,
        callbacks=[cp_callback, log],
        validation_data=val_ds,
    ).history
