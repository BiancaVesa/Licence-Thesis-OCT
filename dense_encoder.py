from keras.applications.densenet import DenseNet121
from keras import layers, models

def get_encoder_dense(image_size, image_channels, code_width, trainable):
    model = DenseNet121(weights="imagenet", include_top=False, input_shape=(image_size, image_size, image_channels))

    # configure model's trainable layers
    if trainable == 0: 
        model.trainable = False
    elif trainable == 1:
        model.trainable = True
    elif trainable == 2:
        layers_dense_trainable = model.layers[:5]
        layers_dense_nontrainable = model.layers[5:]
        for layer_dense in layers_dense_trainable:
            layer_dense.trainable = True
        for layer_dense in layers_dense_nontrainable:
            layer_dense.trainable = False

    batch_norm_layer = layers.BatchNormalization()
    dropout_layer = layers.Dropout(0.2)
    global_avg_pooling = layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(code_width, activation='relu')

    return models.Sequential([
        model,
        batch_norm_layer,
        dropout_layer,
        global_avg_pooling,
        dense_layer
    ], name="encoder")