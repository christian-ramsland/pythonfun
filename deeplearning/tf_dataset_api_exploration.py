import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt # workaround for image viewing in pycharm
import glob
import tensorflow_datasets as tfds
import pandas as pd

from pythonfun.deeplearning.tutorial import plotbert

"""
more like a tutorial on kaggle and loading/preprocessing image data
https://www.tensorflow.org/tutorials/load_data/images

tf.dataset is similar to pandas or numpy except not read into memory.


helpful link for this problem:
https://stackoverflow.com/questions/73701637/how-can-i-create-a-tensorflow-image-dataset-labelled-by-filenames-from-multiple

string splitting:
https://www.w3schools.com/python/ref_string_split.asp
"""


"""
Kaggle stuff:
Retrieved Kaggle's "Dogs vs. Cats" dataset.
package comes from "pip install kaggle"
On kaggle website -> account tab, can download api token "kaggle.json" which can be stored in ./.kaggle/
agree to rules for any given competition
download via "kaggle competitions download -c dogs-vs-cats"
unzip with "unzip "../input/name-of-dataset/test.zip" -d name-of-directory"
"""

"""
square peg round hole with this problem, Amit Dube's reply clarified all the dumb stuff I was doing
https://stackoverflow.com/questions/63121070/tensorflow-binary-image-classification-predict-probability-of-each-class-for-ea
"""


def exploration():
    tfds.list_builders() # interesting set of datasets, will explore at some point but not now.


def get_label(x, label_dict):
    label = x.split('/')[-1].split('.')[0]
    return label_dict[label]


def preprocess(x, y):
  img = tf.io.read_file(x)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, tf.constant([180, 180]))
  # not sure we need this line
  # img = img[..., np.newaxis] # had to do this: https://stackoverflow.com/questions/56874677/transform-3d-tensor-to-4d to get printout x shape= (479, 329, 3, 1) y shape= (2,)
  img = tf.cast(img, dtype=tf.float32)
  # y = tf.convert_to_tensor(y)
  # y = tf.one_hot(y, depth=2)
  # https://stats.stackexchange.com/questions/438875/one-hot-encoding-of-a-binary-feature-when-using-xgboost
  # https://www.tensorflow.org/api_docs/python/tf/one_hot
  # kind of silly to do a one hot encoding since there's only k=2 unique categories
  return img, y

"""
here the test data is not labeled, and the input is of shape [batch_size, image_width, image_height, number_of_channels]
so for predicting an image, need input shape of [1, image_width, image_height, number_of_channels]
"""
def preprocess_test(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, tf.constant([180, 180]))
    img = img[np.newaxis, ...] # knew this would come in handy - actually it didn't I can specify batch size in predict y method
    # actually it did come in handy, specifying batch in predict seems dumb
    return img

def main():
    trainpath_unzipped = os.path.abspath(os.path.expanduser('~') + '/.kaggle/dogs-vs-cats/train.zip')
    # doesn't need to unzip every time so this line is commented out
    # archive_train = tf.keras.utils.get_file(trainpath_unzipped, origin='', extract=True) # unzipped so extract is false - to make this more generalizable set extract to true
    trainpath = os.path.abspath(os.path.expanduser('~') + '/.keras/datasets/train')
    data_dir_train = pathlib.Path(trainpath).with_suffix('')
    image_count_cat = len(list(data_dir_train.glob('cat*.jpg')))
    image_count_dog = len(list(data_dir_train.glob('dog*.jpg')))
    DATASET_SIZE = image_count_cat + image_count_dog
    print(list(data_dir_train.glob('cat*.jpg')))
    label_dict = {'cat': 0, 'dog': 1}
    paths = glob.glob(trainpath + '/*.jpg')

    print(get_label(paths[0], label_dict))
    labels = [get_label(i, label_dict) for i in paths]
    print(labels)

    training_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    training_dataset = training_dataset.shuffle(buffer_size=len(paths)) # amazing resource about exactly this: https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    training_dataset = training_dataset.map(preprocess).batch(batch_size=32)

    for x, y in training_dataset.take(1):
        print("x shape=", x.shape, "y shape=", y.shape)

    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)

    # tf take and shuffle: https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test

    train_dataset = training_dataset.take(train_size)
    val_dataset = training_dataset.skip(train_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 2 # wrong problem

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
        tf.keras.layers.Dense(1, activation='sigmoid') ]) # forgot about this! need sigmoid to be activation function in last layer for binary classification, and it's just one thing

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.binary_crossentropy, # categorical is sparseCategoricalCrossEntropy, binary is binarycrossentropy
        metrics=['accuracy'])

    """
    figure out what's causing this warning though: 
    /home/christian/miniconda3/envs/tf/lib/python3.9/site-packages/keras/backend.py:5585: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
      output, from_logits = _get_logits(
    """

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    # plotbert(history) # need a new plotting method for this kind of problem

    print(model.summary()) # model.summary() with the parens

    testpath_unzipped = os.path.abspath(os.path.expanduser('~') + '/.kaggle/dogs-vs-cats/test1.zip')
    # again don't want to do this twice
    # archive_test = tf.keras.utils.get_file(testpath_unzipped, origin='', extract=True)
    testpath = os.path.abspath(os.path.expanduser('~') + '/.keras/datasets/test1')
    test_paths = glob.glob(testpath + '/*.jpg')

    testing_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    for idx in testing_dataset.take(1):
        print('index shape=', idx.shape, 'index=', idx) # (1, 180, 180, 3) exactly what we wanted to see
    testing_dataset = testing_dataset.map(preprocess_test)
    for idx in testing_dataset.take(1):
        print(idx.shape) # (1, 180, 180, 3) exactly what we wanted to see

    predictions = (model.predict(testing_dataset)) # cats is 0, dogs is 1
    predictions_binarized = np.vectorize(lambda pred: int(pred >= 0.5))(predictions)
    print(predictions_binarized)
    predictions_df = pd.DataFrame.from_records(predictions_binarized, columns=['label'])
    predictions_df['id'] = predictions_df.index + 1 # id starts at 1, index at 0
    predictions_df.insert(0, 'id', predictions_df.pop('id'))
    print(predictions_df.head)


if __name__ == "__main__":
    main()