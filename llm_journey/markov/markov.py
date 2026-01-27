import re
import random
from collections import defaultdict, Counter

class MarkovGenerator:
    """
    A class used to generate text based on the Markov chain algorithm.
    Attributes:
        None
    Methods:
        generate_ngrams(n, txt): Generates n-grams from a given text.
        sanitize_text(input_file_path): Sanitizes a text by removing special characters and converting it to lower case.
        predict_next_word(bigram, vals): Predicts the next word in a sentence based on the last two words.
        generate_sentence(starting_bigram, n, probabilities): Generates a sentence by predicting the next word at each position.    
    """
    def generate_ngrams(self, n, txt):
        """
        This method can be used to generate n-grams

        :param n: N gram
        :param txt: Text to get N-gram
        """
        words = txt.split()

        if len(words) < n:
            return {}
        
        # Generate n-grams
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

        # Count occurrences of each (n-1)-gram and their following word
        context_counts = defaultdict(Counter)
        for ngram in ngrams:
            context, next_word = tuple(ngram[:-1]), ngram[-1]
            context_counts[context][next_word] += 1

        # Calculate probabilities
        probabilities = {}
        for context, next_words_count in context_counts.items():
            total_occurrences = sum(next_words_count.values())
            probabilities[context] = {word: count / total_occurrences for word, count in next_words_count.items()}
        
        return probabilities
    
    def sanitize_text(self, input_file_path):
        """
        This will make everything lower and clean up all special characters

        :param input_file_path: File Path
        """
        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            sanitized_content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
            return sanitized_content
        except FileNotFoundError:
             print("The input file was not found. Please check the file path.")
        except Exception as e:
             print(f"An error occurred: {e}")

        return ""
    
    def predict_next_word(self, bigram, vals):
        """
        Predict the next string based on the last two. At this point we are considering bigrams

        :param bigram: Bigram tuple
        :param vals: All the bigrams
        """
        nextwords = vals.get(bigram)
        if not nextwords:
            return None
        
        words, probs = zip(*nextwords.items())
        word = random.choices(words, weights=probs)[0]
        return word

    def generate_sentence(self, starting_bigram, n, probabilities):
        """
        Keep on prediction to build a sentence

        :param starting_bigram
        :param n: How many words
        :param probabilities
        """
        sentence = list(starting_bigram)
        for _ in range(n - 2):
            current_bigram = tuple(sentence[-2:])
            next_word = self.predict_next_word(current_bigram, probabilities)
            if not next_word:
                break

            sentence.append(next_word)
        return ' '.join(sentence)

    
if __name__ == "__main__":
    mgen = MarkovGenerator()
    santxt = mgen.sanitize_text("../doc/moby.txt").lower()
    probs = mgen.generate_ngrams(3, santxt)
    print(mgen.generate_sentence(("the", "boat"), 50, probs))
