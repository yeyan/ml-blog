#!/bin/bash

# Links to the datasets
links=(
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
)

# Define a variable for data directory,
# in case I want to change the data directory to something else.
data_dir="data"

# Create data directory if not exists
mkdir -p $data_dir

# Download links one by one
for link in "${links[@]}"
do
  wget -P $data_dir $link
done
