import math
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import datetime
import random
import losses
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow import keras
from keras import layers

exec(open("./preprocessing.py").read())
exec(open("./contrastive.py").read())
exec(open("./dense_encoder.py").read())
exec(open("./amd_data.py").read())
exec(open("./evaluate_contrastive.py").read())
exec(open("./acuity_regression.py").read())
exec(open("./acuity_classification.py").read())

unlabeled_dir = "D:/AN4/Licenta/Datasets/AMD/UDF Volume/*/*/*"
labeled_dir = "D:/AN4/Licenta/Datasets/Kermany/OCT2017/train"
classifier_dir = "D:/AN4/Licenta/Datasets/Kermany/OCT2017/test/*/*.jpeg"

# Dataset hyperparameters
unlabeled_dataset_size = 0
labeled_dataset_size = 83484
image_size = 224
image_channels = 3

input_img = layers.Input(shape = (image_size, image_size, image_channels))

# Algorithm hyperparameters
num_epochs = 100
batch_size = 128
width = 128 # Encoder width
temperature = 0.1

# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.75, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}

# Tensorboard logs
log_dir_supervised = "./../out/logs/contrastive_supervised/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_semi = "./../out/logs/contrastive_semi/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_classifier = "./../out/logs/contrastive_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback_supervised = tf.keras.callbacks.TensorBoard(log_dir=log_dir_supervised, histogram_freq=1)
tensorboard_callback_semi = tf.keras.callbacks.TensorBoard(log_dir=log_dir_semi, histogram_freq=1)
tensorboard_callback_classifier = tf.keras.callbacks.TensorBoard(log_dir=log_dir_classifier, histogram_freq=1)

model_checkpoint_path = "D:/AN4/Licenta/Models/contrastive_supervised_smaller_128/checkpoint"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_p_loss', save_weights_only=True, verbose=1, save_best_only=True, mode='min')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_p_loss', min_delta=0.0001, factor = 0.2, patience = 5, verbose = 1, mode='min')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_p_loss', min_delta=0.001, patience=10, verbose=1, mode='min')


def create_model_contrastive(floder_name, is_pretrained):
    model_checkpoint_path = "D:/AN4/Licenta/Models/" + floder_name + "/checkpoint"
    pretraining_model = ContrastiveModel(trainable=1, learning_type='supervised')

    pretraining_model.compile(
        contrastive_optimizer=tf.keras.optimizers.Adam(),
        probe_optimizer=tf.keras.optimizers.Adam(),
    )

    if is_pretrained == True:
        pretraining_model.load_weights(model_checkpoint_path)

    return pretraining_model

# Start model training
def train_model_contrastive(pretraining_model):
    pretraining_history = pretraining_model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset, callbacks=[tensorboard_callback_semi, reduce_lr, model_checkpoint, early_stopping]
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(pretraining_history.history["val_p_acc"]) * 100
        )
    )
    pretraining_model.load_weights(model_checkpoint_path)


def main_function():
    print("Hello user!")
    model_name = input("Type the name of the model you want to evaluate (kermany_classifier/acuity_classifier/acuity_regressor): ")

    if model_name == 'kermany_classifier':
        # Load dataset
        train_dataset, labeled_train_dataset, validation_dataset = prepare_dataset(image_size, image_channels, labeled_dir, unlabeled_dir, labeled_dataset_size, unlabeled_dataset_size)

        folder_name = "contrastive_supervised_smaller_128"
        pretraining_model = create_model_contrastive(folder_name, is_pretrained=True)
        class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        encoder_contrastive = pretraining_model.encoder
        linear_probe_contrastive = pretraining_model.linear_probe

        classifier = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, image_channels)),
                get_augmenter(**classification_augmentation),
                encoder_contrastive,
                linear_probe_contrastive
            ],
            name="classifer",
        )

        print("Evaluating the model, please wait...")
        true_labels, predicted_labels, images = predict_on_test_set(classifier, classifier_dir, image_size, image_size)
        evaluate_model(true_labels, predicted_labels, class_names)

    if model_name == 'acuity_classifier':
        train_dataset, validation_dataset = get_datasets()
        model = get_classifier()
        # train_classifier(model, train_dataset, validation_dataset)
        evaluate_classifier(validation_dataset)

    if model_name == 'acuity_regressor':
        evaluate_regressor()

    # if model_name == "data":
    #     folder_name = "contrastive_supervised_smaller_128"
    #     pretraining_model = create_model_contrastive(folder_name, is_pretrained=True)
    #     encoder_contrastive = pretraining_model.encoder
    #     generate_volumetric_dataset(encoder_contrastive, unlabeled_dir, image_size, 76.334816)


tf.get_logger().setLevel('ERROR')
main_function()
