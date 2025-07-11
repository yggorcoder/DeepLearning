This is just a solution to a proposed challenge:

You must train a model with the IMDB dataset, considering the following assumptions:

The number of words considered in the model must be 1,000.

Truncate or pad the observations so that each has 400 characters.

Add three layers: a 128-dimensional embedding layer; a 128-unit LSTM layer; and a single-unit fully connected layer with a sigmoid activation function.

Build the neural network using the cross-entropy loss function, an Adam-type optimizer, and an accuracy performance metric. Train it for three epochs.
