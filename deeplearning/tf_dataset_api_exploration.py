import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt # workaround for image viewing in pycharm
import glob
import tensorflow_datasets as tfds
import keras.preprocessing.image
from tensorflow.python.data import AUTOTUNE

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


def exploration():
    tfds.list_builders() # interesting set of datasets, will explore at some point but not now.


def get_label(x, label_dict):
    label = x.split('/')[-1].split('.')[0]
    return label_dict[label]


def preprocess(x,y):
  img = tf.io.read_file(x)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.cast(img,dtype=tf.float32)
  return img, y


def main():
    trainpath = os.path.abspath('/home/christian/.kaggle/dogs-vs-cats/train')
    archive_train = tf.keras.utils.get_file(trainpath, origin='', extract=False) # unzipped so extract is false
    data_dir_train = pathlib.Path(archive_train).with_suffix('')
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
    training_dataset = training_dataset.map(preprocess)

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


    num_classes = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    # that's enough for now, hoping to match with https://www.tensorflow.org/tutorials/load_data/images at some point.
    """
    
    print(len(list(data_dir_train.glob('*.jpg'))))
    print(image_count_cat)
    print(image_count_dog)
    labels_list = [image_count_cat, image_count_dog]
    cats = list(data_dir_train.glob('cat*.jpg'))
    dogs = list(data_dir_train.glob('dog*.jpg'))
    img = PIL.Image.open(str(cats[2]))
    plt.imshow(img); plt.show()
    img = PIL.Image.open(str(dogs[1]))
    plt.imshow(img); plt.show()

    batch_size = 32
    img_height = 180
    img_width = 180
    classNames = ['dog', 'cat']

    # tf.keras.utils.image_dataset_from_directory is great for hierarchically labeled directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir_train.glob('*.jpg'),
        validation_split=0.2,
        labels=labels_list,
        label_mode='int',
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    """

if __name__ == "__main__":
    main()