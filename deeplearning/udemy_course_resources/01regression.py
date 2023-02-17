import numpy as np
import tensorflow as tf
import os
import pathlib
import matplotlib.pyplot as plt # workaround for image viewing in pycharm
import glob
import tensorflow_datasets as tfds
import pandas as pd
import tensorflow_hub as hub
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
tf.get_logger().setLevel('ERROR')

def problem1():
    random.seed(0)
    a = np.linspace(start=-10, stop=42, num=100)
    b = np.log(np.linspace(start=0.05, stop=3.84, num=100))

    # a = np.arange(start=-10, stop=42, step=2, dtype=float)
    # b = np.log(np.arange(start=0.25, stop=6.75, step=0.25))
    np.random.seed(0)
    y = (-1.1 * a) + (9 * b) + np.random.normal(0, 3)
    X = pd.DataFrame(a, b)
    X.apply(lambda x: x/x.max(), axis=0)
    plt.plot(a, y)
    plt.plot(b, y)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.mae,
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                            metrics=["MAE", "MSE"])

    history = model.fit(x=X_train, y=y_train, epochs=20, validation_data=(X_val, y_val))

    y_pred = model.predict(X_test)
    model.summary()
    model.evaluate(X_test, y_test)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    plt.scatter(y_test, y_pred)

def problem3():
    random.seed(0)
    insurance_path = os.path.abspath(os.path.expanduser('~') + '/.kaggle/insurance/insurance.csv')
    df = pd.read_csv(insurance_path)
    y = df['charges']
    df = df.drop(['charges'], axis=1)
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
    df = df.drop(['sex_female', 'smoker_no'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15, random_state=0)
    ct = make_column_transformer((MinMaxScaler(), ["age", "bmi", "children"]),
                                 (OneHotEncoder(handle_unknown="ignore"), ["sex_male", "smoker_yes", "region_northeast",
                                                                           "region_northwest", "region_southeast",
                                                                           "region_southwest"]))
    ct.fit(X_train) # fit the column transformer, saves the transformation weights produced by the training set

    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)

    # produce validation data:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.mae,
                  metrics=["MAE"])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)
    results = model.evaluate(X_test, y_test)
    print(model.metrics_names)

    y_pred = model.predict(X_test)
    print(r2_score(y_true=y_test, y_pred=y_pred))
    # this model seems to be the best I've been able to do, this seems to be about what other people get

def problem4():
    tf.random.set_seed(0)
    random.seed(0)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz',
                                                                       test_split=0.2, seed=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    x_test = pd.DataFrame(x_test, columns=columns)
    x_train = pd.DataFrame(x_train, columns=columns)
    x_val = pd.DataFrame(x_val, columns=columns)
    y_train = pd.DataFrame(y_train, columns=['MEDV'])
    y_val = pd.DataFrame(y_val, columns=['MEDV'])
    ct = make_column_transformer((MinMaxScaler(), ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
                                                   'TAX', 'PTRATIO', 'B', 'LSTAT']),
                                 (OneHotEncoder(handle_unknown="ignore"), ['CHAS', 'RAD']))
    ct.fit(x_train)
    x_train = ct.transform(x_train)
    x_val = ct.transform(x_val)
    x_test = ct.transform(x_test)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mae',
                  metrics=['MAE'])
    model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val), epochs=50)

    y_pred = model.predict(x_test)
    model.evaluate(x_test, y_test)
    print(r2_score(y_true=y_test,y_pred=y_pred))



def classification_silliness():
    iris = datasets.load_iris()
    features = pd.DataFrame(iris.data, columns = iris.feature_names)
    print(features.head)
    labels = pd.DataFrame(iris.target)

def main():
    print('no error')

# problem 1: test r^2 at ~ 0.97 which seems reasonable
# problem 2: test r^2 at ~0.96 which is still quite good
# problem 3: 3a) 4 dense layers didn't seem to work too well for me
# 3b) adding more hidden units seemed to work quite well
# 3c) the first parameter is the learning rate, you get a substantially better model in my circumstances if lr incr.
# 3d) 300 epochs seems to be its own kind of over-training

if __name__ == '__main__':
    problem1()
    problem3()
    problem4()

