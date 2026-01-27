import nltk
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter

class DataProcessor:
    """
    A class used to process and prepare data for training.
    Attributes:
        file_path (str): The path to the file containing the data.
        corpus (list): A list of words extracted from the text.
        word_to_idx (dict): A dictionary mapping each word to its corresponding index.
        idx_to_word (dict): A dictionary mapping each index to its corresponding word.
    Methods:
        load_data: Loads data from the specified file path and tokenizes it into words.
        build_vocab: Builds a vocabulary of unique words in the corpus and creates the word-to-index and index-to-word dictionaries.
        encode: Encodes the text by replacing each word with its corresponding index.
    Notes:
        The downloaded 'punkt' package is required for word tokenization.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = []
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def load_data(self):
        """
        Loads data from the specified file path.

        This method reads the text from the file, tokenizes it into words,
        and stores the result in the `corpus` attribute of the class instance.
        
        Note: The downloaded 'punkt' package is required for word tokenization.
        """
        with open(self.file_path, 'r') as file:
            text = file.read()[:10000]
            nltk.download('punkt')
            self.corpus = word_tokenize(text.lower())
    
    def build_vocab(self):
        """
        Builds the vocabulary of the corpus by counting the occurrences of each word.

        Returns:
            None
        """
        counter = Counter(self.corpus)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(counter.items())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def encode(self) -> list:
        """
        Returns a list of encoded indices corresponding to words present in the corpus.

        The encoding is done by mapping each word to its respective index in the word-to-index 
        dictionary.
        """
        return [self.word_to_idx[word] for word in self.corpus if word in self.word_to_idx]
    
    def save_dict(self, outfile):
        """
        Creates a dictionary mapping each word to its corresponding index.

        Returns:
            dict: A dictionary containing the word-to-index mapping.
        """
        encoding_dict = {word: idx for idx, word in enumerate(self.word_to_idx)}
        with open(outfile, 'wb') as f:
            pickle.dump(encoding_dict, f)

    def load_dict(self, infile):
        with open(infile, 'rb') as f:
            encoding_dict = pickle.load(f)
        self.word_to_idx = encoding_dict
        self.idx_to_word = {value: key for key, value in encoding_dict.items()}
        

if __name__ == "__main__":
    dp = DataProcessor("../doc/moby.txt")
    #dp.load_data()
    #dp.build_vocab()
    #dp.save_dict('./z_corpus.pk')
    dp.load_dict("./z_corpus.pk")
    print(dp.word_to_idx)
    #print(dp.idx_to_word[20000])
