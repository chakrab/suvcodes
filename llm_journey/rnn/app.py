
import torch
from dataprocessor import DataProcessor
from rnnmodel import RNNModel
from rnntrainer import RnnTrainer
from tester import Tester

class App:
    """
    App class representing the main application of the RNN model.

    This class handles the initialization and training of the RNN model,
    as well as generating text based on input sequences.

    Attributes:
        device (torch.device): The device to use for computations (mps or cpu).
        embedding_dim (int): The dimensionality of the word embeddings.
        hidden_dim (int): The dimensionality of the hidden state.
        vocab_size (int): The size of the vocabulary.
        model (RNNModel): The trained RNN model.
        trainer (RnnTrainer): The trainer object for training the model.
        tester (Tester): The tester object for generating text.

    Methods:
        main(): The main entry point of the application, responsible for loading data,
            building the vocabulary, initializing the model and trainer, training the
            model, and generating text.
    """
    @staticmethod
    def save_model(model, file_path):
        """
        Save model
        """
        torch.save(model.state_dict(), file_path)
        print("Model saved successfully.")
    
    @staticmethod
    def load_model(model_path, device):
        """
        Load Model
        """
        model = RNNModel(vocab_size=7680, embedding_dim=128, hidden_dim=256).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    @staticmethod
    def main():
        """
        The main entry point of the application.

        This method is responsible for loading data, building the vocabulary,
        initializing the model and trainer, training the model, and generating text.
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Step 1
        print("Start...", flush=True)
        file_path = '../doc/moby.txt'
        data_processor = DataProcessor(file_path)
        data_processor.load_data()
        data_processor.build_vocab()
        encoded_data = data_processor.encode()
        print("Built Vocab...", flush=True)

        # Step 2: Model Parameters
        embedding_dim = 128
        hidden_dim = 256
        vocab_size = len(data_processor.word_to_idx)
        print(f"Model Vocab size: {vocab_size}", flush=True)

        # Step 3: Initialize Model, Trainer, and Tester
        model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)
        seq_length = 30  # Length of the input sequence for training
        trainer = RnnTrainer(model, encoded_data, seq_length, epochs=5)
        
        # Step 4: Train the Model
        print("Start Training...", flush=True)
        trainer.train(device)
            
        # Test
        start_string = "love of"
        tester = Tester(model, data_processor.idx_to_word, data_processor.word_to_idx)  # Add word_to_idx
        generated_text = tester.generate_text(start_string, gen_length=5, device=device)
        print("Generated Text:")
        print(generated_text)

if __name__ == "__main__":
    App.main()

"""
Start Training...
Epoch 1/5, Loss: 4.377009868621826, time (s): 0.8220899105072021
Epoch 2/5, Loss: 2.6552248001098633, time (s): 0.5460391044616699
Epoch 3/5, Loss: 1.3255549669265747, time (s): 0.5451779365539551
Epoch 4/5, Loss: 0.663205087184906, time (s): 0.544252872467041
Epoch 5/5, Loss: 0.38068661093711853, time (s): 0.5430550575256348
Next Word: oil
Next Word: swimming
Next Word: in
Next Word: the
Next Word: ocean
Generated Text:
love of oil swimming in the ocean
"""