import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, models
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns

MODEL_NAME = "./data/volumetric_model_2cls.hdf5"
IMG_SIZE = (32, 128)


def get_datasets():
    BATCH_SIZE = 32

    train_dir = 'D:\AN4\Licenta\Datasets\AMD_encoded_classes_img_2cls'

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                color_mode="grayscale",
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                validation_split=0.2,
                                                                seed=123,
                                                                subset="training",
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                color_mode="grayscale",
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                validation_split=0.2,
                                                                seed=123,
                                                                subset="validation",
                                                                image_size=IMG_SIZE)

    return train_dataset, validation_dataset

def get_classifier():
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(500,kernel_regularizer=regularizers.l2(0.01),activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(2)
    ])

    return model

def train_classifier(model, train_dataset, validation_dataset):

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min')

    epochs = 100
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[ checkpoint]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(100)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def evaluate_classifier(test_dataset):
    model = tf.keras.models.load_model(MODEL_NAME)

    predictions = model.predict(test_dataset)
    true_labels = np.concatenate([true_labels for x, true_labels in test_dataset], axis=0)

    predicted_categories = np.array([])

    for prediction in predictions:
        score = tf.nn.softmax(prediction)
        predicted_categories = np.concatenate([predicted_categories, [np.argmax(score)]])

    print(classification_report(true_labels, predicted_categories))
    cm = confusion_matrix(true_labels, predicted_categories)
    print(cm)
    sns.heatmap(cm, cmap="Blues", annot=True)