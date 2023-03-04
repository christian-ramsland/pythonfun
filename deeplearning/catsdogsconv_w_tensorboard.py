import tensorflow as tf
import os
import pathlib
import glob
from time import time
import tensorflow_hub as hub
from kagglecatsvsdogsconvNN import get_label, preprocess
import numpy as np
import pandas as pd

tf.random.set_seed(12)
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
tf.get_logger().setLevel('ERROR')
# run this in shell beforehand: tensorboard --logdir=logs/

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def main():
    trainpath_unzipped = os.path.abspath(os.path.expanduser('~') + '/.kaggle/dogs-vs-cats/train.zip')
    # doesn't need to unzip every time so this line is commented out
    archive_train = tf.keras.utils.get_file(trainpath_unzipped, origin='', extract=True) # unzipped so extract is false - to make this more generalizable set extract to true
    trainpath = os.path.abspath(os.path.expanduser('~') + '/.keras/datasets/train')
    data_dir_train = pathlib.Path(trainpath).with_suffix('')
    image_count_cat = len(list(data_dir_train.glob('cat*.jpg')))
    image_count_dog = len(list(data_dir_train.glob('dog*.jpg')))
    DATASET_SIZE = image_count_cat + image_count_dog
    # print(list(data_dir_train.glob('cat*.jpg')))
    label_dict = {'cat': 0, 'dog': 1}
    paths = glob.glob(trainpath + '/*.jpg')

    # print(get_label(paths[0], label_dict))
    labels = [get_label(i, label_dict) for i in paths]
    # print(labels)

    training_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    training_dataset = training_dataset.map(preprocess).batch(batch_size=16)

    for x, y in training_dataset.take(1):
        print("x shape=", x.shape, "y shape=", y.shape)

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(training_dataset,
                                                          ds_size=tf.data.experimental.cardinality(training_dataset).numpy(),
                                                          shuffle=True)

    # tf take and shuffle: https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test
    # addendum for TF 2.9 https://stackoverflow.com/questions/64305438/warning-the-calling-iterator-did-not-fully-read-the-dataset-being-cached-in-or


    AUTOTUNE = tf.data.AUTOTUNE
    numEpochs = 10

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'), # it's not happy with this layer
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') ])


    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))
    tensorboard_eval = tf.keras.callbacks.TensorBoard(log_dir='logs/evaluate/')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.binary_crossentropy, # categorical is sparseCategoricalCrossEntropy, binary is binarycrossentropy
        metrics=['accuracy'])
    tf.random.set_seed(12)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=numEpochs,
        callbacks=[tensorboard]
    )

    predictions = model.predict(test_ds)
    model.evaluate(test_ds, callbacks=[tensorboard_eval])
    predictions_binarized = np.vectorize(lambda pred: int(pred >= 0.5))(predictions)
    predictions_df = pd.DataFrame.from_records(predictions_binarized, columns=['label'])
    predictions_df['id'] = predictions_df.index + 1 # id starts at 1, index at 0
    predictions_df.insert(0, 'id', predictions_df.pop('id'))
    predictions_df.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    main()