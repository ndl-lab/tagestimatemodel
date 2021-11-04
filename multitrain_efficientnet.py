from tensorflow.keras.utils import Sequence,to_categorical
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import BinaryCrossentropy
#from tensorflow.compat.v1.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import os
import tensorflow as tf
class DatasetSequence(Sequence):
    def __init__(self, x_set, y_set,batch_size, augmentations):
        self.x=x_set
        self.batch_size = batch_size
        self.augment = augmentations
        self.y = np.array(y_set)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([cv2.imread(file_name,1) for file_name in batch_x]) 
        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)

import cv2
from albumentations import (
    Compose, HorizontalFlip,VerticalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,RandomResizedCrop,Resize,
    ToFloat, ShiftScaleRotate,ToGray
)

AUGMENTATIONS_TRAIN = Compose([
    Resize(224,224,p=1),
    ToGray(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.2),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    RandomResizedCrop(224,224,scale=(0.9,1.1),p=1),
    ToFloat(max_value=255)
])

AUGMENTATIONS_TEST = Compose([
    Resize(224,224,p=1),
    ToFloat(max_value=255)
])

train_paths=[]
train_classes_onehot=[]
classdic={}
import glob
import json
import numpy as np
np.random.seed(777)
for path in glob.glob("datasetpdm/*/*"):
    classname=path.split(os.sep)[1]
    train_paths.append(path)
    vector=np.zeros(11)
    if classname=="graphic_illust":
        vector[0]=1
    elif classname=="graphic_graph":
        vector[0]=1
        vector[2]=1
    elif classname=="graphic_map":
        vector[0]=1
        vector[3]=1
    elif classname=="graphic_illustcolor":
        vector[0]=1
        vector[4]=1
    elif classname=="picture_indoor":
        vector[1]=1
        vector[5]=1
    elif classname=="picture_object":
        vector[1]=1
        vector[6]=1
    elif classname=="picture_person":
        vector[1]=1
        vector[7]=1
    elif classname=="picture_outdoor":
        vector[1]=1
        vector[8]=1
    elif classname=="picture_landmark":
        vector[1]=1
        vector[8]=1
        vector[9]=1
    elif classname=="stamp":
        vector[10]=1
    train_classes_onehot.append(vector)

idx = np.random.permutation(len(train_paths))
train_paths=np.array(train_paths)[idx].tolist()
train_classes_onehot=np.array(train_classes_onehot)[idx].tolist()

val_count = int(len(train_paths) * 0.1)
train_gen = DatasetSequence(
                             train_paths[val_count:], 
                             train_classes_onehot[val_count:],
                             batch_size=48, augmentations=AUGMENTATIONS_TRAIN)
val_gen = DatasetSequence(
                             train_paths[:val_count], 
                             train_classes_onehot[:val_count],
                             batch_size=48, augmentations=AUGMENTATIONS_TEST)

base_model =EfficientNetB0(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='linear', activity_regularizer=None)(x)
x = Dense(512, activation = 'relu')(x)
predictions = Dense(11, activation = 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)
print(model.summary())
weights_dir = './weights_pdm_multi/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)

model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    period = 1
)
class MultiLabelMacroF1(tf.keras.metrics.Metric):
    def __init__(self, name='multi_label_macro_f1', threshold=0.5, **kwargs):
        super(MultiLabelMacroF1, self).__init__(name=name, **kwargs)
        #self.specificity = self.add_weight(name='mlm_spec', initializer='zeros')
        self.f1score = self.add_weight(name='mlm_spec', initializer='zeros')
        self.threshold       = tf.constant(threshold)
        # replace this with tf confusion_matrix utils
        self.true_positives  = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    def update_state(self, y_true, y_pred):
        # Compare predictions and threshold.
        pred_is_pos  = tf.greater(tf.cast(y_pred, tf.float32), self.threshold)
        pred_is_neg  = tf.logical_not(tf.cast(pred_is_pos, tf.bool))
        # |-- in case of soft labeling
        label_is_pos = tf.greater(tf.cast(y_true, tf.float32), self.threshold)
        label_is_neg = tf.logical_not(tf.cast(label_is_pos, tf.bool))
        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(pred_is_pos, label_is_pos), tf.float32))
        )
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(pred_is_pos, label_is_neg), tf.float32))
        )
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(pred_is_neg, label_is_pos), tf.float32))
        )
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        precision=tf.math.divide_no_nan(tp, tf.add(tp, fp))
        recall=tf.math.divide_no_nan(tp, tf.add(tp, fn))
        f1score=tf.math.divide_no_nan(tf.multiply(tf.constant(2,tf.float32),tf.multiply(precision,recall)),tf.add(precision,recall))
        self.f1score.assign(f1score)
        return f1score
    def result(self):
        return self.f1score


model.compile(
    optimizer = Adam(0.0001),
    loss = BinaryCrossentropy(from_logits=False),
    metrics = [MultiLabelMacroF1()]
)

model.fit_generator(
           train_gen, 
           validation_data=val_gen, 
           epochs=100,
           shuffle=True,callbacks=[model_checkpoint])
