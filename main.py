 # -*- coding:utf-8 -*- 
import numpy as np
import tensorflow as tf

import math
import os
import sys
import time
import data_helper
from model import Seq2SeqModel_Train, Seq2SeqModel_Infer
from trainer import Trainer
from evaluator import Evaluator

def main(args):
    with tf.device("/gpu:1"):
        print('Load data files...')

        print('*' * 10 + ' Train')
        train_data = data_helper.load_data(args, 'train', 20000)
        print('*' * 10 + ' Test')
        test_data = data_helper.load_data(args, 'test')

        print('-' * 50)
        print('Build dictionary..')

        args.word_dict_en = data_helper.build_dict(train_data.data[0] + test_data.data[0])
        args.word_dict_zh = data_helper.build_dict(train_data.data[1] + test_data.data[1])

        print('-' * 50)

        args.embeddings_en = data_helper.gen_embeddings(args.word_dict_en, args.embedding_size_en, args.embedding_file_en)
        args.embeddings_zh = data_helper.gen_embeddings(args.word_dict_zh, args.embedding_size_zh, args.embedding_file_zh)
        (args.encoder_vocab_size, args.embedding_size_en) = args.embeddings_en.shape
        (args.decoder_vocab_size, args.embedding_size_zh) = args.embeddings_zh.shape

        train_data.vectorize(args.word_dict_en, args.word_dict_zh)
        test_data.vectorize(args.word_dict_en, args.word_dict_zh)
        

        with tf.variable_scope('root'):
            model_train = Seq2SeqModel_Train(args)
            initializer = tf.global_variables_initializer()
        with tf.variable_scope('root', reuse=True):
            model_infer = Seq2SeqModel_Infer(args)
        

        trainer = Trainer(args, model_train)
        evaluator = Evaluator(args, model_infer)

        timestamp = str(int(time.time()))
        out_dir = os.path.join(args.out_dir, timestamp)
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)        
        saver = tf.train.Saver(tf.global_variables())
        
        config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config_gpu.gpu_options.allow_growth = True
        sess = tf.Session(config=config_gpu)
        sess.run(tf.global_variables_initializer())

        

        # Training
        print('-' * 50)
        print('Start training..')
        start_time = time.time()
        last_time = start_time        
        n_updates = 0
        batch100_time = 0
        best_bleu = 0
        for epoch in range(args.num_epoches):
            for idx, batch in enumerate(train_data.gen_minbatches(args.batch_size, shuffle=True)):
                train_loss, train_op = trainer.step(sess, batch)
                batch_time = time.time() - last_time
                if idx % 20 == 0:
                    print('Epoch = %d, iter = %d, loss = %.2f, batch time = %.2f (s)' %
                             (epoch, idx, train_loss, batch_time))
                n_updates += 1
                batch100_time = batch100_time + batch_time
                # Evalution
                if n_updates % args.eval_iter == 0:

                    train_bleu = evaluator.get_evaluation(sess, train_data.gen_minbatches(args.batch_size, start_examples=8000, end_examples=10000))
                    print('Epoch = %d, iter = %d, train_bleu = %.2f' % (epoch, idx, train_bleu))

                    test_bleu = evaluator.get_evaluation(sess, test_data.gen_minbatches(args.batch_size))
                    print('Epoch = %d, iter = %d, test_bleu = %.2f' % (epoch, idx, test_bleu))

                    if test_bleu > best_bleu:
                        best_bleu = test_bleu
                        print('Best test bleu: epoch = %d, n_udpates = %d, bleu = %.2f ' % (epoch, n_updates, test_bleu))

                last_time = time.time()
