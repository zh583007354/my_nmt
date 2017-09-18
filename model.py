#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class Seq2SeqModel_Train(object):
    """docstring for Seq2SeqModel"""
    def __init__(self, args):
        self.args = args

        # inputs
        self.encoder_inputs = tf.placeholder('int32', shape=[None, None], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder('int32', shape=[None, None], name='decoder_inputs')
        self.decoder_outputs = tf.placeholder('int32', shape=[None, None], name='decoder_outputs')
        

        self.enc_mask = tf.placeholder('bool', shape=[None, None], name='enc_mask')
        self.dec_mask = tf.placeholder('bool', shape=[None, None], name='dec_mask')
        self.is_inference = tf.placeholder('bool', shape=[], name='is_inference')
        
        self.logits = None
        self.translations = None
        self.loss = None

        self._build_forward()
        self._build_loss()


    def _build_forward(self):
        args = self.args

        VS_enc = args.encoder_vocab_size
        VS_dec = args.decoder_vocab_size
        HS = args.hidden_size
        BS = tf.shape(self.encoder_inputs)[0]
        maximum_iterations = tf.shape(self.encoder_inputs)[1]

        start_tokens = tf.fill([BS], 1)
        end_token = 2
    
        # embeddings
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            enc_embedding = tf.Variable(args.embeddings_en, dtype='float', name='enc_embedding', trainable=False)
            dec_embedding = tf.Variable(args.embeddings_zh,  dtype='float', name='dec_embedding', trainable=False)
        # lookup
        with tf.variable_scope('lookup'):
            enc_input = tf.nn.embedding_lookup(enc_embedding, self.encoder_inputs)
            dec_input = tf.nn.embedding_lookup(dec_embedding, self.decoder_inputs)

        enc_len = tf.reduce_sum(tf.cast(self.enc_mask, 'int32'), 1)
        dec_len = tf.reduce_sum(tf.cast(self.dec_mask, 'int32'), 1)

        # encoder
        with tf.variable_scope('encoder'):
            enc_cell = tf.contrib.rnn.BasicLSTMCell(HS)
            enc_cell_wrap = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=args.dropout_rate, output_keep_prob=args.dropout_rate)
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=enc_cell_wrap,
                                                       inputs=enc_input,
                                                       sequence_length=enc_len,
                                                       dtype='float')
                                                       # time_major=True)
        # tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(args.sos)),
        #                  tf.int32)
        # tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(args.eos)),
        #                  tf.int32)
        # start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        # end_token = tgt_eos_id

        # helper
        with tf.variable_scope('helper'):
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input, sequence_length=dec_len)
            # if not self.is_inference:
            #     helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input, sequence_length=dec_len)# time_major=True
            # else:
            #     helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embedding, start_tokens, end_token)

        # decoder
        with tf.variable_scope('decoder'):
            projection_layer = layers_core.Dense(VS_dec, use_bias=False, name="projection_layer")
            
            dec_cell = tf.contrib.rnn.BasicLSTMCell(HS)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                      helper=helper,
                                                      initial_state=enc_state,
                                                      output_layer=projection_layer)

            f_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

            logits = f_outputs.rnn_output

            # if not self.is_inference:
            #     f_outputs, f_state, f_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
            #     logits = f_outputs.rnn_output
            #     translations = None
            # else:
            #     f_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=args.maximum_iterations)
            #     logits = None
            #     translations = f_outputs.sample_id
            
            self.logits = logits

    def _build_loss(self):
        with tf.variable_scope('loss'):
            loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                    targets=self.decoder_outputs,
                                                    weights=tf.cast(self.dec_mask, 'float'),
                                                    name='loss')
            self.loss = loss
            # crossen = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_y, logits=logits)
            # train_loss = (tf.reduce_sum(crossen * self.dec_mask) / batch_size)
    
    def get_loss(self):
        return self.loss



class Seq2SeqModel_Infer(object):
    """docstring for Seq2SeqModel"""
    def __init__(self, args):
        self.args = args

        # inputs
        self.encoder_inputs = tf.placeholder('int32', shape=[None, None], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder('int32', shape=[None, None], name='decoder_inputs')
        self.decoder_outputs = tf.placeholder('int32', shape=[None, None], name='decoder_outputs')
        

        self.enc_mask = tf.placeholder('bool', shape=[None, None], name='enc_mask')
        self.dec_mask = tf.placeholder('bool', shape=[None, None], name='dec_mask')
        self.is_inference = tf.placeholder('bool', shape=[], name='is_inference')
        
        self.logits = None
        self.translations = None
        self.loss = None

        self._build_forward()


    def _build_forward(self):
        args = self.args

        VS_enc = args.encoder_vocab_size
        VS_dec = args.decoder_vocab_size
        HS = args.hidden_size
        BS = tf.shape(self.encoder_inputs)[0]
        maximum_iterations = tf.shape(self.encoder_inputs)[1]

        start_tokens = tf.fill([BS], 1)
        end_token = 2
    
        # embeddings
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            enc_embedding = tf.Variable(args.embeddings_en, dtype='float', name='enc_embedding', trainable=False)
            dec_embedding = tf.Variable(args.embeddings_zh, dtype='float', name='dec_embedding', trainable=False)
        # lookup
        with tf.variable_scope('lookup'):
            enc_input = tf.nn.embedding_lookup(enc_embedding, self.encoder_inputs)
            dec_input = tf.nn.embedding_lookup(dec_embedding, self.decoder_inputs)

        enc_len = tf.reduce_sum(tf.cast(self.enc_mask, 'int32'), 1)
        dec_len = tf.reduce_sum(tf.cast(self.dec_mask, 'int32'), 1)

        # encoder
        with tf.variable_scope('encoder'):
            enc_cell = tf.contrib.rnn.BasicLSTMCell(HS)
            enc_cell_wrap = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=args.dropout_rate, output_keep_prob=args.dropout_rate)
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=enc_cell_wrap,
                                                       inputs=enc_input,
                                                       sequence_length=enc_len,
                                                       dtype='float')
                                                       # time_major=True)
        # tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(args.sos)),
        #                  tf.int32)
        # tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(args.eos)),
        #                  tf.int32)
        # start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        # end_token = tgt_eos_id

        # helper        
        with tf.variable_scope('helper'):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embedding, start_tokens, end_token)
            # if not self.is_inference:
            #     helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input, sequence_length=dec_len)# time_major=True
            # else:
            #     helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embedding, start_tokens, end_token)

        # decoder
        with tf.variable_scope('decoder'):
            projection_layer = layers_core.Dense(VS_dec, use_bias=False, name="projection_layer")
            
            dec_cell = tf.contrib.rnn.BasicLSTMCell(HS)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                      helper=helper,
                                                      initial_state=enc_state,
                                                      output_layer=projection_layer)

            f_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)

            translations = f_outputs.sample_id

            # if not self.is_inference:
            #     f_outputs, f_state, f_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
            #     logits = f_outputs.rnn_output
            #     translations = None
            # else:
            #     f_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=args.maximum_iterations)
            #     logits = None
            #     translations = f_outputs.sample_id
            
            self.translations = translations

    def get_translations(self):
        return self.translations
