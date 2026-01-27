import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    Class representing a dataset for text data.
    Attributes:
        encoded_data (list): The list of encoded text data.
        seq_length (int): The length of the sequence to be used for training.
    Methods:
        __len__: Returns the number of sequences available in the dataset.
        __getitem__: Returns a tuple containing the input and output tensors for a given index.
    """
    def __init__(self, encoded_data, seq_length):
        self.encoded_data = encoded_data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.encoded_data) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.encoded_data[idx: idx + self.seq_length]), 
                torch.tensor(self.encoded_data[idx + 1: idx + self.seq_length + 1]))

class RnnTrainer:
    """
    A class used to train an RNN model.
    Attributes:
        model (nn.Module): The RNN model to be trained.
        seq_length (int): The sequence length of the input data.
        batch_size (int, optional): The batch size for training. Defaults to 64.
        epochs (int, optional): The number of epochs for training. Defaults to 10.
        criterion (nn.CrossEntropyLoss): The loss function used for training.
        optimizer (optim.Adam): The optimizer used for training the model.
        train_loader (DataLoader): The data loader for the training dataset.
    Methods:
        train(device): Trains the model on the given device.
    """
    def __init__(self, model, train_data, seq_length, batch_size=64, epochs=10, lr=0.001):
        self.model = model
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_loader = DataLoader(TextDataset(train_data, seq_length), batch_size=batch_size, shuffle=True)
    
    def train(self, device):
        """
        Trains the RNN model on the training dataset.

        Args:
            device (torch.device): The device to use for training (e.g., Metal or CPU).

        Returns:
            None
        """
        for epoch in range(self.epochs):
            start = time.time()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
            end = time.time()
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}, time (s): {end - start}')
