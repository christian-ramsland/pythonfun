"""
answers to: https://github.com/mrdbourke/tensorflow-deep-learning
"""

import tensorflow as tf

#1
scaler = tf.constant([10])
vector = tf.constant([10, 14])
matrix = tf.constant([[11, 17], [13, 2]])
tensor = tf.constant([[[1, 9], [2, 8]], [[21, 6], [9, 3]]])
#2
def findattributes(tensortype, x):
    print(tensortype)
    print('shape=', x.shape)
    print('rank', x.ndim)
    print('size', tf.size(x))

findattributes('scaler', scaler)
findattributes('vector', vector)
findattributes('matrix', matrix)
findattributes('tensor', tensor)

#3
random_tensor = tf.random.uniform(shape=(5, 300))
random_tensor2 = tf.random.uniform(shape=(5, 300))
findattributes('random tensor', random_tensor)

#4
# random_tensor @ random_tensor2 doesn't work obviously Matrix size-incompatible: In[0]: [5,300], In[1]: [5,300] [Op:MatMul]
print(tf.linalg.matmul(random_tensor, tf.transpose(random_tensor2))) # or some variation of this would work

# 5
tf.tensordot(random_tensor, random_tensor2, axes=0)

#6
image_tensor = tf.random.uniform(shape=(224, 224, 3))
#7
tf.reduce_min(image_tensor, axis=0) #
tf.reduce_max(image_tensor, axis=1)

#8
extra_dim_image_tensor = tf.random.normal(shape=[1, 224, 224, 3])
extra_dim_image_tensor = tf.squeeze(extra_dim_image_tensor)
print(extra_dim_image_tensor.shape)

#9
tf.random.set_seed(42)
vector = tf.random.uniform(shape=[10], minval=0, maxval=200, seed=42)
print(vector[:])
print('min position=', tf.argmin(vector))
print('min value', vector[tf.argmin(vector)])

# one hot encode the tensor in 9
depth = vector.get_shape().as_list()[0]
print(depth)
tf.cast(vector, dtype=tf.int32)
vector
indices = vector.numpy()
print(indices)
one_hot_vector = tf.one_hot(tf.cast(vector, dtype=tf.int32), depth=10)
one_hot_vector

# this is the right answer no idea why I'm not seeing more 1's in my one-hot
ten_tensor = tf.constant([0 ,1 ,2 ,3, 4, 9, 10, 0.1 , -000.1 , 6])
tf.cast(ten_tensor , dtype=tf.int32)
tf.one_hot(tf.cast(ten_tensor , dtype=tf.int32) , depth = 10)