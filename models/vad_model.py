import tensorflow as tf
import numpy as numpy
from basic_files.vad import *
from basic_files.rnn_cell import *
import sys


class BasicAttention:

    """ Class Defines the basic attention model : 
        as defined in Paper : A neural attention model for abstractive text summarization
    """ 

    def add_cell(self,hidden_size, cell_input=None):

        """ Define the rnn_cell to be used in attention model

            Args:
                cell_input: Type of rnn_cell to be used. Default: LSTMCell
                hidden_size : Hidden size of cell
        """

        if(cell_input is None):
            self.enc_cell  = GRUCell(hidden_size)
        else:
            self.enc_cell = cell_input


    def add_projectionLayer(self, hidden_size, len_vocab):

        """ Add the projection layer for hidden_size x vocab

            Args:
                hidden_size : The hidden size of the cell
                len_vocab   : The number of symbols in vocabulary
        """
        self.projection_B = tf.get_variable(name="Projection_B", shape=[len_vocab])
        self.projection_W = tf.get_variable(name="Projected_W", shape=[hidden_size, len_vocab])


    def inference(self, encoder_inputs1, decoder_inputs1, query_inputs, embedding_size, feed_previous,
                  len_vocab, hidden_size, weights, embedding_trainable,  initial_embedding = None, c=None):

        """ Builds the graph for the basic attetion model

            Args:
                encoder_inputs: Placeholder for the encoder sequence
                decoder_inputs: Placeholder for the decoder sequence
                query_inputs  : Placeholder for the query   sequence
                embedding_size: Dimensions of the embedding for encoder and decoder symbols.
                feed_previous : Boolean to decide whether to feed previous state output 
                                to current rnn state for the decoder.
                len_vocab     : Number of symbols in encoder/decoder.
                hidden_size   : Hidden size of the cell state
                weights       : The weights given to the decoder labels, accordingly 
                                which will contribute to the loss calculation
                c             : The cell that needs to be used.
        
            Returns:
                A list of tensors of size [batch_size * num_symbols], that gives the
                probability distribution over symbols for each time step. The list
                is of size max_sequence_length
        """


        self.add_cell(hidden_size, c)
        self.add_projectionLayer(hidden_size, len_vocab)

        cell_encoder_bw = GRUCell(hidden_size)
        ei = tf.unpack(encoder_inputs1)
        di = tf.unpack(decoder_inputs1)
        outputs, state     = vad_seq2seq(encoder_inputs = ei,
                                         decoder_inputs = di,
                                         cell_encoder_fw = self.enc_cell,
                                         cell_encoder_bw = cell_encoder_bw,
                                         embedding_trainable=embedding_trainable,
                                         num_encoder_symbols= len_vocab,
                                         num_decoder_symbols= len_vocab,
                                         embedding_size = embedding_size,
                                         output_projection= (self.projection_W, self.projection_B),
                                         feed_previous= feed_previous,
                                         initial_embedding = initial_embedding,
                                         dtype=tf.float32)

        self.final_outputs = [tf.matmul(o, self.projection_W) + self.projection_B for o in outputs]

        return self.final_outputs


    def loss_op(self, outputs, labels, weights):

        """ Calculate the loss from the predicted outputs and the labels

            Args:
                outputs : A list of tensors of size [batch_size * num_symbols]
                labels : A list of tensors of size [sequence_length * batch_size]
                weights: Weights given to the decoder labels to compute the loss

            Returns:
                loss: loss of type float
        """

        _labels = tf.unpack(labels)
        weights = tf.to_float(weights)
        _weights = tf.unpack(weights)
        #print(_weights[0].get_shape())
        loss_per_batch = sequence_loss(outputs, _labels, _weights)

        self.calculated_loss =  loss_per_batch
        return loss_per_batch


    def training(self, loss, learning_rate):

        """ Creates an optimizer and applies the gradients to all trainable variables.

            Args:
                loss : Loss value passed from function loss_op
                learning_rate : Learning rate for GD.

            Returns:
                train_op : Optimizer for training
        """

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op