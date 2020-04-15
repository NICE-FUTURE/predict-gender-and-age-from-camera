from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from data import Data
# from custom_metrics import *

batch_size = 256
epochs = 150

# CNN
def train_CNN(x_train, y_train, x_test, y_test):
    input_layer = Input(shape=(20, 20, 1), name="input_layer")

    conv2d_layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    conv2d_layer = Conv2D(64, (3, 3), activation='relu')(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    pool_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(pool_layer)
    hidden_layer = Dense(128, activation='relu')(flatten_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    output_layer = Dense(units=1, activation="relu", name="output_layer")(hidden_layer)
    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2)
    return model, history
   

def visualize_history(history):

    plt.subplot("111")
    plt.plot(history["loss"], c="blue", label="loss")
    plt.plot(history["val_loss"], c="orange", label="val_loss")
    plt.legend()

    plt.savefig("train4age_history.png")

if __name__ == "__main__":
    data = Data(trainpath="./data/trainset/", testpath="./data/testset/")
    x_train, y_train, x_test, y_test = data.load4age()

    x_train = x_train.reshape(x_train.shape[0], 20, 20, 1)
    x_test = x_test.reshape(x_test.shape[0], 20, 20, 1)

    model, history = train_CNN(x_train, y_train, x_test, y_test)
    model.save("cnn4age.h5")
    visualize_history(history.history)
