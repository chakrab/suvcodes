import torch.nn as nn

class RNNModel(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) model for sequence prediction tasks.
    It consists of an embedding layer, an RNN cell, and a fully connected layer.
    Attributes:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the embedded words.
        hidden_dim (int): The number of units in the hidden state.
    Methods:
        forward(x): The forward pass through the network. It takes an input tensor x and returns a predicted output tensor.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
