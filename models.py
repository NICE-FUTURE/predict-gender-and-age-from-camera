from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Flatten
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16


def cnn_regression(height, width, channel):
    input_layer = Input(shape=(height, width, channel), name="input_layer")

    conv2d_layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    conv2d_layer = Conv2D(64, (3, 3), activation='relu')(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    pool_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(pool_layer)
    hidden_layer = Dense(128, activation='relu')(flatten_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    output_layer = Dense(units=1, name="output_layer")(hidden_layer)

    model = Model(input_layer, output_layer)
    return model


def cnn_classification(height, width, channel):
    input_layer = Input(shape=(height, width, channel), name="input_layer")

    conv2d_layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    conv2d_layer = Conv2D(64, (3, 3), activation='relu')(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    pool_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(pool_layer)
    hidden_layer = Dense(128, activation='relu')(flatten_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    output_layer = Dense(units=1, activation="sigmoid", name="output_layer")(hidden_layer)

    model = Model(input_layer, output_layer)
    return model
   

def vgg16_regression(height, width, channel):
    inputs = Input(shape=(height, width, channel))

    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    for layer in base_model.layers:
        layer.trainable = False

    x = Conv2D(512, kernel_size=(3,3), activation="relu")(base_model.output)
    x = Conv2D(128, kernel_size=(1,1), activation="relu")(x)
    x = Conv2D(1, kernel_size=(1,1))(x)
    outputs = Flatten()(x)

    model = Model(inputs, outputs)
    with open("age_model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s:print(s, file=f))
    return model


def vgg16_classification(height, width, channel):
    inputs = Input(shape=(height, width, channel))

    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    for layer in base_model.layers:
        layer.trainable = False

    x = Conv2D(512, kernel_size=(3,3), activation="relu")(base_model.output)
    x = Conv2D(128, kernel_size=(1,1), activation="relu")(x)
    x = Conv2D(1, kernel_size=(1,1), activation="sigmoid")(x)
    outputs = Flatten()(x)

    model = Model(inputs, outputs)
    with open("gender_model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s:print(s, file=f))
    return model
