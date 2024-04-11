Here We made the for loop of functional calls of the model (distil bert) successufully.

There are bugs, in which the model only predicts one of the two classes (0 or 1).

This is may be, because we are using old code , as implemented in the official examples of FOSI repository:

[FOSI-Repository](https://github.com/hsivan/fosi/blob/main/examples/fosi_torch_train_cifar10_with_batch_norm.py)

The main.py file is not updated, only the .ipynb