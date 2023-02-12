import numpy as np
import tensorflow as tf
import os
import pathlib
import matplotlib.pyplot as plt # workaround for image viewing in pycharm
import glob
import tensorflow_datasets as tfds
import pandas as pd
import tensorflow_hub as hub


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
tf.get_logger().setLevel('ERROR')


if __name__ == "__main__":
    print('hello world')