 #-*- coding:utf-8 -*-
import os
import tensorflow as tf
from main import main as m

flags = tf.app.flags

flags.DEFINE_string("mode", "train", "train | validation | test | forward [test]")
flags.DEFINE_string("data_dir", "data", "Data dir [data]")
flags.DEFINE_string("out_dir", "out", "out dir [out]")
flags.DEFINE_string("embedding_file_en", "data/glove.6B.100d.txt", "embedding_file_en")
flags.DEFINE_string("embedding_file_zh", "data/wiki.ch.text.vector", "embedding_file_zh")

flags.DEFINE_integer("batch_size", 10, "Batch size [60]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate for Adam [0.001]")
flags.DEFINE_integer("num_epoches", 50, "Total number of epochs for training [50]")
flags.DEFINE_integer("eval_num_batches", 100, "eval num batches [100]")
flags.DEFINE_integer("embedding_size_en", 100, "embedding size of English [100]")
flags.DEFINE_integer("embedding_size_zh", 300, "embedding size of Chinese [300]")
flags.DEFINE_integer("eval_iter", 100, "evaluation per x steps [100]")

flags.DEFINE_integer("sos", 1, "start token <s> [1]")
flags.DEFINE_integer("eos", 2, "end token </s> [2]")

flags.DEFINE_boolean("load", False, "load saved data? [True]")
flags.DEFINE_boolean("is_inference", False, "is_inference? [False]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_float("dropout_rate", 0.8, "Input keep prob [0.8]")
flags.DEFINE_float("grad_clipping", 5.0, "gradients clipping [5.0]")



def main(_):
    args = flags.FLAGS

    m(args)

if __name__ == "__main__":
    tf.app.run()
