#-*- coding:utf-8 -*-

import tensorflow as tf

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.loss = model.get_loss()
        self.opt = tf.train.AdamOptimizer(args.init_lr)
        self.grads = self.opt.compute_gradients(self.loss)
        # clip_grads = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in self.grads]
        self.train_op = self.opt.apply_gradients(self.grads)
        # self.summary = model.summary

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {self.model.encoder_inputs : batch[0],
                     self.model.decoder_inputs : batch[1],
                     self.model.decoder_outputs : batch[2],
                     self.model.enc_mask : batch[3],
                     self.model.dec_mask : batch[4],
                     self.model.is_inference : False}
        loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        # logits = sess.run(self.model.logits, feed_dict=feed_dict)
        # print(batch[2])
        # print(logits.shape)
        # exit()
        return loss, train_op


