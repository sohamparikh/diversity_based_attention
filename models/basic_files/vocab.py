# Vocab() class is defined here.
 
import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer 
from gensim.models import Word2Vec
from collections import defaultdict
from math import sqrt
import numpy as np 
import matplotlib.pyplot as plt 

class Vocab():

    def __init__(self):

        """ Initalize the class parameters to default values
        """

        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown       = "<unk>"
        self.end_of_sym    = "<eos>"
        self.start_sym     = "<s>"
        self.padding       = "<pad>"
        self.word_freq     = {}
        self.len_vocab     = 0
        self.total_words   = 0
        self.embeddings    = None


    def get_global_embeddings(self, filenames, embedding_size):

        """ Construct the Embedding Matrix for the sentences in filenames.

            Args:
                filenames: File names of the training files: Based on 
                which the vocab will be built. This is used when there
                are no pretrained embeddings present. Then instead of 
                using random embeddings, Word2Vec algorithm is used 
		to train the embeddings on the dataset avaliable.
                embedding_size: Dimensions for the embedding to be used.

            Returns
                Embedding matrix.
        """
        sentences = []

        if (os.path.exists('../Data/embeddings.bin') == True):
            model = Word2Vec.load_word2vec_format('../Data/embeddings.bin', binary = True)
	    print ("Loading pretriained embeddings")
        else:
            for file in filenames:
                with open(file, 'rb') as f:
                    for lines in f:
                        words = [lines.split()]
                        sentences.extend(words)

            model = Word2Vec(sentences, size=embedding_size, min_count=0)
            model.save("../Data/embeddings.bin")

        self.embeddings_model = model
        return model

    def add_constant_tokens(self):

        """ Adds the tokens <pad> and <unk> to the vocabulary.
        """

        self.word_to_index[self.padding]    = 0
        self.word_to_index[self.unknown]    = 1


    def add_word(self, word):

        """ Adds the word to the dictionary if not already present. 

        Arguments:
             word : Word to be added.

        Returns:
            * void
        """
        if word in self.word_to_index:
            self.word_freq[word] = self.word_freq[word] + 1

        else:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.word_freq[word] = 1
        
    def create_reverse_dictionary(self):

        """ Creates a mapping from index to the words
            This will be helpful in decoding the predictions
            to sentences.
        """

        for key, val in self.word_to_index.iteritems():
            self.index_to_word[val] = key

    def construct_dictionary_single_file(self, filename):
        
        """ Adds the words belonging to this file to the
            dictionary

            Arguments:
                * filename: The respective file from which words
		  needs to be added.
            Returns:
                * void
        """
        with open(filename, 'rb') as f:
            for lines in f:
                for words in lines.split():
                    self.add_word(words)


    def fix_the_frequency(self, limit=0):

        """ Eliminates the words from the dictionary with 
	    a frequency less than the limit provided in the 
	    argument.

	    Arguments:
                * limit: The threshold frequency

            Returns:
		* void
        """

        temp_word_to_index = {}
        temp_index_to_word = {}

        word_list = []
        count = 0

        #
        # Start from index 2 so that the constant tokens:
	# <pad> and <unk> are not eliminated
	#
        new_index = 2
        for key in self.word_to_index:
            if (self.word_freq[key] > limit):
                temp_word_to_index[key] = new_index
                temp_index_to_word[new_index] = key
                new_index  = new_index + 1

        self.word_to_index = temp_word_to_index


    def construct_dictionary_multiple_files(self, filenames):

        """ Dictionary is made from the words belonging to all
            the files in the set filenames

            Arguments :
                * filenames = List of the filenames 

            Returns :
                * None
        """

        for files in filenames:
            self.construct_dictionary_single_file(files)


    def encode_word(self, word):

        """ Convert the word to the particular index

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]


    def decode_word(self, index):

        """ Index is converted to its corresponding word.

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_word:
            return self.unknown
        return self.index_to_word[index]


    def get_embeddings(self, embedding_size):

        """ This function creates an embedding matrix
            of size (vocab_size * embedding_size). The embedding 
	    for each word is loaded from the embeddings learnt in the 
            function get_global_embeddings(). 

            Arguments:
		* embedding_size: Dimension size to represent the word.

            Returns:
		* void
        """

        sorted_list = sorted(self.index_to_word.items(), key = operator.itemgetter(0))
        embeddings = []

        np.random.seed(1357)

        for index, word in sorted_list:

            if word in self.embeddings_model.vocab:
                embeddings.append(self.embeddings_model[word])
            else:
                if word in ['<pad>', '<s>', '<eos>']:
                    temp = np.zeros((embedding_size))
                else:
                    temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size),
                                             (embedding_size))
                embeddings.append(temp)

        self.embeddings = np.asarray(embeddings)
        self.embeddings = self.embeddings.astype(np.float32)

    def construct_vocab(self, filenames, embedding_size):


        """ Constructs the vocab, and initializes the embeddings 
            accordingly 

            Args:
                * filenames: List of filenames required to generate the vocab
                * embeddings: Dimension size for the word representation

            Returns:
                * void
        """

        self.get_global_embeddings(filenames, embedding_size)
        self.construct_dictionary_multiple_files(filenames)
        self.fix_the_frequency(0)
        self.add_constant_tokens()
        self.create_reverse_dictionary()
        self.get_embeddings(embedding_size)

        self.len_vocab = len(self.word_to_index)

        print ("Number of words in the vocabulary is " + str(len(self.word_to_index))

        self.total_words = float(sum(self.word_freq.values()))
