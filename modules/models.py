import torch
import torch.nn as nn


class NNModel(nn.Module):
    """
    A subclass of the `Module` object representing a feed-forward neural
    network.

    Note: this is exaclty the same as subclassing a `Layer` or `Model` object
          in TensorFlow.
    """
    def __init__(self):
        """
        Class constructor.
        """
        # Call the parent class' constructor.
        super().__init__()

        # Define the layers (i.e. submodules, which are `Module` objects
        # themselves) as attributes of the class.
        self.linear_1 = nn.Linear(1, 10)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(10, 20)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(20, 10)
        self.relu_3 = nn.ReLU()
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        """
        Define the forward pass for the model given the input `x`.
        """
        out = self.relu_1(self.linear_1(x))
        out = self.relu_2(self.linear_2(out))
        out = self.relu_3(self.linear_3(out))
        out = self.output_layer(out)

        return out


class FFNN(nn.Module):
    """
    Subclass of the `Module` object representing an arbitrary feed-forward NN
    with the specified number of hidden layers (and their dimension) and
    activation function.
    """
    def __init__(
            self,
            dims,
            activation='relu',
            output_activation=nn.Identity()
        ):
        """
        Class constructor. `dims` is a list of int representing the
        dimension of each layer (the first being the input dimension, while
        the last is the output one).
        """
        super().__init__()

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=dims[i], out_features=dims[i+1])
            for i in range(len(dims) - 1)
        ])

        if activation.lower() == 'relu':
            self.activations = nn.ModuleList([
                nn.ReLU()
                for _ in range(len(dims) - 1)
            ])
        else:
            raise NotImplementedError(
                f'Support for activations {activation} not implemented'
            )

        self.activations.append(output_activation)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        out = x

        for layer, activation in zip(self.linear_layers, self.activations):
            out = activation(layer(out))

        return out
