"""
Question 1: What are the key terms? e.g. explain convolution in your own words, pooling in your own words

for a 2D convolution, the input tensor and conv2d output tensor consist of two spatial dimensions (width & height), and
one feature dimension (rgb color for images in image input).

The kernel or filter is a 4d tensor that stores weights and biases used for recognizing "patterns" of the layer,
whether that's an edge or line earlier on in the network or the makeup of a face or vehicle later on. Each of the kernel
weights correspond to a region of the input, and when the region of a particular filter is multiplied by the corresponding
weight/bias of the filter, the output value is some number that varies depending on how well the input matched some
pattern of a given class.

the CNN explainer site seems to regard the relu activation layer as worth highlighting just as prominently as the conv
layer, and while nonlinearity is important for differentiation between classes since you don't want the class
probability prediction to simply be some linear combination of the inputs, I don't think it's as interesting except that
it sort of acts to emphasize the fact that the output is just another 3D tensor, a better approximation below:
https://www.youtube.com/watch?v=eMXuk97NeSI&t=254s

Feature map/activation map/rectified feature map all mean the same exact thing, it's called an activation map because it
is a mapping that corresponds to the activation of different parts of the image.

The pooling layer is responsible for 'blurring' the spatial extent of the network, so a 9x9 region could become a 1x1
or similar, it reduces the # of parameters used later on in the network. This also helps to reduce overfitting, the
inclusion of maxpooling2D layers reduced the # of parameters by 50x and led to an improved validation score.

The flatten layer simply removes any spatial organization of a feature map. A 4x4x10 3D tensor becomes a vector with
160 values in it.
"""