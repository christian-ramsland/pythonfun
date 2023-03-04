import tensorflow as tf
import os
import pathlib
import glob
from time import time
import tensorflow_hub as hub
from kagglecatsvsdogsconvNN import get_label, preprocess
from catsdogsconv_w_tensorboard import get_dataset_partitions_tf

tf.random.set_seed(12)
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
tf.get_logger().setLevel('ERROR')

class input_pipeline:
    train_ds = None
    val_ds = None
    test_ds = None
    batch_size = None

    def __init__(self):
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
        batch_size = 16
        self.batch_size = batch_size

        training_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        training_dataset = training_dataset.map(preprocess).batch(batch_size)

        for x, y in training_dataset.take(1):
            print("x shape=", x.shape, "y shape=", y.shape)

        self.train_ds, self.val_ds, self.test_ds = get_dataset_partitions_tf(training_dataset,
                                                              ds_size=tf.data.experimental.cardinality(training_dataset).numpy(),
                                                              shuffle=True)

def compare_learning_rates(pipeline, initial_learning_rate, final_learning_rate):

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = pipeline.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = pipeline.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    numEpochs = 10
    train_size = tf.data.experimental.cardinality(train_ds).numpy()

    # exponential learning rate decay
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / numEpochs)
    steps_per_epoch = int(train_size / pipeline.batch_size)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

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

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{initial_lr}{time}'.format(initial_lr=initial_learning_rate, time=time()))
    tensorboard_eval = tf.keras.callbacks.TensorBoard(log_dir='logs/evaluate/')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.binary_crossentropy, # categorical is sparseCategoricalCrossEntropy, binary is binarycrossentropy
        metrics=['accuracy'])
    tf.random.set_seed(12)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=numEpochs,
        callbacks=[tensorboard]
    )

    model.evaluate(pipeline.test_ds, callbacks=[tensorboard_eval])



if __name__ == "__main__":
    pipeline = input_pipeline()
    compare_learning_rates(pipeline, 0.0011, 0.0009)