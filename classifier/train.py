import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "../flask/model.h5"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
NUM_KEYWORDS = 10


def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and outputs
    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return x, y


def get_data_splits(data_path, test_size=0.1, validation_size=0.1):

    # load the dataset
    x, y = load_dataset(data_path)

    # create train/validation/test splits
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # convert inputs from 2d to 3d arrays
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):

    # build network
    model = tf.keras.models.Sequential()

    # conv layer 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # conv layer 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # conv layer 3
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten the output and feed to dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(tf.keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))

    # compile the model
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


def main():

    # load train/ validation/ test data splits
    x_train, x_validation, x_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN Model
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(x_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test error: {test_error}, Test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
