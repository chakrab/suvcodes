import torch

class Tester:
    """
    Class Tester provides methods for generating text based on a given model and vocabulary.
    Attributes:
        model (nn.Module): The neural network model used for text generation.
        idx_to_word (dict): Mapping of integer indices to words.
        word_to_idx (dict): Mapping of words to integer indices.
    Methods:
        generate_text(start_string, gen_length=100, device=None):
            Generates a specified length of text starting from the given start string.
    """
    def __init__(self, model, idx_to_word, word_to_idx):
        self.model = model
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
    
    def generate_text(self, start_string, gen_length=100, device=None):
        """
        Method to generate text

        :param start_string: Start of string
        :param gen_length: How many words to generate
        :param device: MPS or CPU? since I am on MacBook
        """
        self.model.eval()
        generated_text = start_string
        
        # Convert start_string to token indices
        input_eval = torch.tensor([self.word_to_idx[word] 
                            for word in start_string.split() 
                            if word in self.word_to_idx]).unsqueeze(0).to(device)

        for _ in range(gen_length):
            predictions = self.model(input_eval)
            predicted_id = torch.argmax(predictions[-1]).item()

            if predicted_id in self.idx_to_word:
                print(f"Next Word: {self.idx_to_word[predicted_id]}")
                generated_text += ' ' + self.idx_to_word[predicted_id]
            else:
                print(f"Warning: predicted_id {predicted_id} not in idx_to_word map. Skipping.")
                break
            
            # Update input_eval with the new predicted_id for the next iteration
            input_eval = torch.cat((input_eval[:, 1:], torch.tensor([[predicted_id]]).to(device)), dim=1)
            
        return generated_text
