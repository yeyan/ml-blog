---
title: "Deep Learning with PyTorch (Part 1)"
date: "2020-12-09"
markup: pdc
---

Neural network is one of the hottest topics today. Scientists have invented lots of specialized variations of them: Long/Short Term Memory (LSTM), Gated Recurrent Unit (GRU), Deep Convolutional Network (DCN), Auto Encoder (AE) and etc. All of those are essentially different ways of constructing parametric functional approximations. In this post, I would like to introduce some basics about neural networks.


### Neural Network to the Bare-Bones

The simplest neural network actually has its own name: Logistic Regression. Assuming we have an input vector $X$ and a parametric matrix $M$, then it can be expressed as $\hat{Y}=\sigma(MX)$ where $\sigma$ is the sigmoid function which is $\sigma(x) = 1/(1 + e^{-x})$. A plain fully connected deep neural network can be expressed as the following:

$$
\begin{aligned}
A_0 &= X
\\
A_1 &= h(M_1A_0)
\\
A_2 &= h(M_2A_1)
\\
A_3 &= h(M_3A_2)
\\
\vdots
\\
\hat{Y} &= h(M_{n}A_{n})
\end{aligned}
$$

$X$ is the input, $Y$ is the output. $h$ here is a non-linear function which takes a vector as input and outputs another vector, theoretically it can be any function that is  differentiable. In practice, we often pick functions like ReLU, LeakyReLU or sigmoid (Ironically, ReLU and LeakyReLU is not differentiable at 0, but we still use them as they have computation advantages). $M_1, M_2 \dots,M_n$ are parametric matrices between each layer.

### MNIST

It is possible for one to hand make a neural network. But in practice, we often choose a mature deep learning framework to start with. Strictly speaking all those deep learning frameworks are essentially auto differentiation engines. As the raise of CUDA, all the main stream deep learning framework provide GPU based calculation, which is often a few times faster than CPU based calculation.

In this post, we choose PyTorch as our deep learning framework. I find PyTorch hides less than Tensorflow in API level, which make it easier when you want to use some uncanny architecture. In the rest of this post we will use PyTorch to train a neural network that can recognize MNIST hand writing digits.

#### Download Training and Testing Data

MNIST data is available almost everywhere, lots of deep learning framework ships MNIST dataset as a part of the package. In this post, we will do it in the old fashion way: download MNIST from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/). I have made a simple shell script to download the dataset which can be found [here](download.sh).

#### Load and Clean Up Data

The data was code in IDX file format. The first 4 bytes are the magic number of the file. Depends on the dimension of the array, the consecutive bytes (4 as a chunk) indicates the shape of the array stored. The following code shows how to convert the binary file into an numpy array.

{{<highlight py>}}
def load_data(filename):
    """
    Load data from Idx file.
        The first 4 bytes are magic number which indicates the file type
        The following bytes indicates the shape of the data
    """
    # determine the dimensionality of the array.
    match = re.search(re.compile('idx(?P<index_size>\d+)-ubyte.gz$'), filename)
    if not match:
        raise InvalidArgument("Not a idx byte file!")

    with gzip.open(filename, 'r') as fd:
        # parse the shape of the array.
        shape = [int.from_bytes(fd.read(4), 'big') for i in range(int(match.group('index_size')) + 1)][1:]
        # load data and reshape as the file indicated.
        return np.frombuffer(fd.read(), dtype=np.uint8).reshape(shape)
{{</highlight>}}

Load data as numpy array is not the end of the story. The pixel value is ranged from 0 to 255, feed this directly to neural network will cause the parameters of the neural network flutter up and down violently, therefore we normalize it by dividing 255. Further more, as PyTorch only operates on tensors, all the data needs to convert into tensors.



{{<highlight py>}}
# If GPU is available, we will prefer GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def prepare(x, y):
    """
    Convert numpy arrays to tensors, and also normalize images.
    """
    # normalize image pixels
    images = torch.from_numpy(x.astype(np.float32))
    images /= 255.
    images = images.unsqueeze(1)

    # move data to GPU if available
    return images.to(device), torch.from_numpy(y).to(device, dtype=torch.long)

# Load training data.
train_x, train_y = prepare(
  load_data('data/train-images-idx3-ubyte.gz'),
  load_data('data/train-labels-idx1-ubyte.gz')
)

# Load testing data.
test_x, test_y = prepare(
    load_data('data/t10k-images-idx3-ubyte.gz'),
    load_data('data/t10k-labels-idx1-ubyte.gz')
)
{{</highlight>}}

In the above code, the planed output of neural network is in one-hot format. Theoretically, $y$ here need to be in one-hot format as well. But PyTorch provides a very convenient loss functor named `CrossEntropyLoss` which has done all those for us. Therefore, we only need to pack the ground truth label in a LongTensor.

#### Describe The Neural Network Architecture in PyTorch

The following code shows a "deep" convolutional network, that reads an 28x28 grey scale hand writing digit and outputs the probabilities of being each digits (0 to 9):

{{<highlight py>}}
import torch
from torch import nn

class Model(nn.Module):
    """
    Convolutional neural network for MNIST dataset
    """
    def __init__(self):
        # Required for any PyTorch Module
        super().__init__()

        # Layers
        self.layers = nn.ModuleList([
            # Convolutional Layer 1
            nn.Conv2d(1, 32, 3, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.MaxPool2d(2),
            # Convolutional Layer 2
            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.MaxPool2d(2),
            # Convolutional Layer 3
            nn.Conv2d(64, 128, 3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.MaxPool2d(2),
            # Convolutional Layer 4
            nn.Conv2d(128, 128, 3, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            # Fully Connected layer 1
            nn.Linear(1152, 512),
            nn.ReLU(),
            # Fully Connected layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            # Fully Connected layer 3
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        ])
        # Move model to GPU if possible
        self.to(device)

    def forward(self, x):
        # Apply each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x
{{</highlight>}}

*Convolutional Layer*

Convolutional layers are very common in computer vision, as they shine in capturing local features. Assuming $X$ is a 3x3 matrix which represents a channel of a image, and K is a 2x2 matrix which is a kernel/filter matrix. Then their convolution can be express as the following:

$$
\begin{aligned}
X &= \begin{bmatrix}
  x_1 & x_2 & x_3\\
  x_4 & x_5 & x_6\\
  x_8 & x_7 & x_9\\
\end{bmatrix}
\\
K &= \begin{bmatrix}
  k_1 & k_2\\
  k_3 & k_4\\
\end{bmatrix}
\\
Conv(X, K) &= \begin{bmatrix}
  x_1k_1 + x_2k_2 + x_4k_3 + x_5k_4 & x_2k_1 + x_3k_2 + x_5k_3 + x_6k_4\\
  x_4k_1 + x_5k_2 + x_8k_3 + x_7k_4 & x_5k_1 + x_6k_2 + x_7k_3 + x_9k_4
\end{bmatrix}
\end{aligned}
$$

We can also illustrate the idea in Python:

{{<highlight py>}}
from torch.nn import functional as F

def conv_channel(inputs, kernel, padding=(0, 0), stride=(1, 1)):
    iheight, iwidth = inputs.shape
    pheight, pwidth = padding

    # pad input image with zeros around border
    padded_inputs = torch.zeros(iheight + 2 * pheight, iwidth + 2 * pwidth)
    padded_inputs[pheight:pheight + iheight, pwidth: pwidth + iwidth] = inputs

    kheight, kwidth = kernel.shape

    # calculate output shape
    output_shape = (iheight - kheight + 1 + 2 * pheight, iwidth - kwidth + 1 + 2 * pwidth)
    output_shape = [np.ceil(o / s).astype(int) for o, s in zip(output_shape, stride)]

    # initialize output tensor
    output = torch.zeros(output_shape)

    for h in range(output_shape[0]):
        for w in range(output_shape[1]):
            # calculate with part of input image we want to filter
            hi, wi = [d * s for d, s in zip([h, w], stride)]
            # apply kernel/filter matrix to the part of the image and assign the result to output
            output[h, w] = (kernel * padded_inputs[hi:hi + kheight, wi:wi + kwidth]).sum()

    return output

# Define a single 1 channel (grey scale) 5x5 image
# Batch Size, Channel, Height, Width
inputs = torch.randn(1, 1, 5, 5)

# Define a single 2x3 kernel/filter
# Output Channel, Input Channel, Height, Width
kernel = torch.randn(1, 1, 2, 3)

# Compare conv_channel's output and PyTorch's conv2d's output
assert torch.eq(
    conv_channel(inputs[0, 0, :, :], kernel[0, 0, :, :], padding=(1, 2), stride=(5, 3)),
    F.conv2d(inputs, kernel, padding=(1, 2), stride=(5, 3))[0, 0, :, :]
).all().item()
{{</highlight>}}

To be continued ...
