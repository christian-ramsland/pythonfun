import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_moons
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix



# I didn't have to do any of this hahahaha

# have to adapt some javascript code to our own purposes in order to do the first problem.
# in playground.ts, NUM_SAMPLES_CLASSIFY = 500 so that's the first function parameter we know
# and according to state.ts, noise is default = 0 which can be confirmed visually by the application
#     let r = randUniform(0, radius * 0.5);
# radius is 5 for outer circle plus noise factor, inner circle radius is 2.5 plus a noise factor
#
"""
export function classifyCircleData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}
"""

def prettyConfusionMatrix(y_test, y_preds, classes):
    import itertools
    figsize = (10, 10)
    # create the confusion matrix
    cm = confusion_matrix(y_test, y_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize
    n_classes = cm.shape[0]

    # make it pretty
    fig, ax = plt.subplots(figsize = figsize)
    # create the matrix legend
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # label axes
    ax.set(title="confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)


    # set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=15)

    # set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    return plt


def getCircleLabel(p, radius):
    return (1 if np.linalg.norm(p) < (radius * 0.5) else -1)

def gendata():
    numSamples = int(500)
    radius = 5
    noise = 0
    # generate positive points in the circle
    data = pd.DataFrame(columns = ['x', 'y', 'label'])
    for i in range(numSamples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle) # little refresher on sine and cosine lol: https://stackoverflow.com/questions/3488569/why-is-cosine-used-to-calculate-the-x-values-and-sine-the-y-values-for-an-arc
        y = r * np.cos(angle) # I always remember y being associated with sin but w/e
        noiseX = np.random.uniform(-radius, radius) * noise
        noiseY = np.random.uniform(-radius, radius) * noise
        point = [x + noiseX, y + noiseY]
        label = getCircleLabel(point, radius)
        # print(label)
        data.loc[i, 'x'] = point[0]
        data.loc[i, 'y'] = point[1]
        data.loc[i, 'label'] = label
        # prints all 1s here which is intended
    for k in range(numSamples // 2):
        df_index = k + numSamples // 2
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noiseX = np.random.uniform(-radius, radius) * noise
        noiseY = np.random.uniform(-radius, radius) * noise
        point = [x + noiseX, y + noiseY]
        label = getCircleLabel(point, radius)
        data.loc[df_index, 'x'] = point[0]
        data.loc[df_index, 'y'] = point[1]
        data.loc[df_index, 'label'] = label

    # ok things are looking good here
    # print(data.head)
    # print(data.tail)
    data['x'] = data['x'].astype(np.float32)
    data['y'] = data['y'].astype(np.float32)
    return data

def problem2():
    data = gendata()
    dummies = pd.get_dummies(data['label'], drop_first=False, dtype=np.int64)
    df = pd.concat([data, dummies], axis=1)
    df.drop(df.iloc[:, 2:4], axis=1, inplace=True)
    df = df.rename({df.columns[2] : "label"}, axis=1)
    df = df.astype({col: 'int32' for col in df.select_dtypes('int64').columns})
    train_data, test_data = train_test_split(df, test_size=0.5, shuffle=True)
    train_labels = train_data['label']
    test_labels = test_data['label']
    train_data = train_data.drop(['label'],axis=1)
    test_data = test_data.drop(['label'],axis=1)


    """
    print(train_data.head)
    print('train labels head=', train_labels.head)
    print('train labels tail=', train_labels.tail)
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=100)
    print('evaluation time:')
    model.evaluate(test_data, test_labels)
    model.summary()

def problem3():
    X, y = make_moons(n_samples=150)
    print(y[:])
    features = pd.DataFrame(X, columns=['X1', 'X2'])
    labels = pd.DataFrame(y, columns=['label'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()
    data = pd.DataFrame({'X1':X[:, 0], 'X2':X[:, 1], 'label':y[:]})
    print(data.head)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=50)
    model.evaluate(X_test, Y_test)

def plotimageandlabel(index, train_data, train_labels, class_names):
    plt.imshow(train_data[index])
    plt.title(class_names[train_labels[index]])

def preprocess(x):
    x[..., np.newaxis]
    return x


def problem4():
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    plt.imshow(train_data[0])
    train_data = train_data[..., np.newaxis]
    test_data = test_data[..., np.newaxis]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_count = len(np.unique(test_labels))
    plotimageandlabel(30, train_data, train_labels, class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(class_count, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=15)

    model.evaluate(test_data, test_labels)
    predictions = model.predict(test_data)
    top_k_values, top_k_indices = tf.nn.top_k(predictions)
    y_preds = np.array(top_k_indices)
    """
    indices = np.transpose(np.where(y_pred != 0))
    np.where and np.transpose are pretty helpful: https://stackoverflow.com/questions/21887138/iterate-over-the-output-of-np-where
    unfortunately in my case what happened was that only 9200 had probabilities > 0.5 so I had to round
    """
    plot = prettyConfusionMatrix(test_labels, y_preds, None)
    plot.show()

    # no point in doing this redundant pipeline here
    problem6(y_preds, test_labels, test_data, class_names)

def problem5(tensor):
    tensor2 = vectorized_softmax(tensor)
    tensor3 = tf.nn.softmax(tensor)
    np.testing.assert_array_equal(tensor2.numpy(), tensor3.numpy())
    return 0
def vectorized_softmax(tensor):
    tensor2 = np.exp(tensor) / np.sum(np.exp(tensor))
    return tf.constant(tensor2, dtype=tf.float32)

def softmax_normal_loop(tensor):
    denom_sum = 0
    for i in range(tensor.shape[-1]):
        # print(tensor[i].numpy())
        denom_sum = denom_sum + np.exp(tensor[i].numpy())
    # print('denom_sum=', denom_sum)
    list = []
    for i in range(tensor.shape[-1]):
        list.append(np.exp(tensor[i].numpy()) / denom_sum)
    return tf.constant(np.array(list))

def problem6(y_preds, test_labels, test_data, class_names):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))
    # axs.set_xticks(ticks, minor=False)
    image_indices = np.array([0,1,72], dtype=np.int32)
    for idx, item in enumerate(image_indices):
        plot_image(image_indices[idx], y_preds[item], test_labels, test_data, class_names, axs[idx])
    fig.show()

def plot_image(i, predictions_array, true_label, img, class_names, ax):
  true_label, img = true_label[i], img[i]
  ax.grid(False)
  ax.set(xticks=([]))
  ax.set(yticks=([])) #, yticks([])

  ax.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  ax.set(xlabel = "{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]))
  return 0


def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def problem7(test_labels, class_name, class_names, train_data):
    class_idx = class_names.index(class_name)
    all_instances_of_class = [i for i in range(len(test_labels)) if test_labels[i] == class_idx]
    pick_three = all_instances_of_class[:3]
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))
    fig.suptitle(class_name)
    for idx, item in enumerate(pick_three):
        # print('pick three index in this loop:', pick_three[idx])
        img = train_data[pick_three[idx]]
        axs[idx].grid(False)
        axs[idx].set(xticks=([]))
        axs[idx].set(yticks=([]))
        axs[idx].imshow(img, cmap=plt.cm.binary)
    fig.show()


if __name__ == '__main__':
    tensor = tf.constant([16, 27, 78, 99, 189], dtype=tf.float64)
    problem5(tensor)
    tensor2 = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
    problem5(tensor2)

    # whatever close enough






