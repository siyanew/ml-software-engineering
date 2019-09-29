import os
import numpy as np
from typing import List
import word2vec

from data.commit import Commit

train_file = "./model_data/commit_msgs_train.txt"
model_output_file = './model_data/trained_test_embedding.bin'
word_vector_size = 10


class TextEmbedder:

    def __init__(self):
        pass

    def embed(self, msg: str) -> np.array:
        """Generates a vector for a message by appending the word2vec vectors of the words in that message"""
        if not os.path.exists(model_output_file):
            print("No word2vec model trained yet. Run the 'train_embedding' function first.")
            return

        model = word2vec.load(model_output_file)
        message_vector = np.array([])
        for word in msg.lower().split(' '):
            if word not in model.vocab:
                # Maybe do something smarter than ignoring?
                print(f"'{word}' is not in the vocabulary of this model and will be ignored")
                continue

            word_vector = model[word]
            message_vector = np.append(message_vector, word_vector)

        print(f" Message Vector: {message_vector}")
        return message_vector

    def train_embedding(self, commit_list_train: List[Commit]):
        # Populate the training file
        with open(train_file, "w") as text_file:
            for commit in commit_list_train:
                print(commit.msg.lower(), file=text_file)

        # We can experiment with the word vector size and min-count values here
        # as well as using phrases as input instead of words using 'word2phrase'
        word2vec.word2vec(train_file, model_output_file, size=word_vector_size, verbose=True, min_count=0)


# Example usage:
commit_list = [Commit('', '', '', 'Fixed a bug regarding blabla'), Commit('', '', '', 'Cleaned up the code')]
text_embedder = TextEmbedder()
text_embedder.train_embedding(commit_list)
text_embedder.embed('Fixed a bug and cleaned up the code')
