#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import bleu

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.dict = {v : k for k, v in  args.word_dict_zh.items()}
        self.translations = self.model.get_translations()

    def get_evaluation(self, sess, batches):
        bleu_score = 0
        for idx, batch in enumerate(batches):
            feed_dict = {self.model.encoder_inputs : batch[0],
                         self.model.decoder_inputs : batch[1],
                         self.model.decoder_outputs : batch[2],
                         self.model.enc_mask : batch[3],
                         self.model.dec_mask : batch[4],
                         self.model.is_inference : True}
            translation_ids = sess.run(self.translations, feed_dict=feed_dict)
            reference_ids = batch[2]
            reference_lengths = batch[4]
            translations = []
            references = []
            for i in range(len(batch)):
                translation = []
                for j in range(len(translation_ids[i])):
                    translation.append(self.dict[translation_ids[i][j]])
                    # print(translation)
                translations.append("".join(translation))
                # print(translations)
                reference = []
                for k in range(int(np.sum(reference_lengths[i]))):
                    reference.append(self.dict[reference_ids[i][k]])
                    # print(reference)
                references.append(["".join(reference)])
                # print(references)
                # exit()
            # print(translations)
            # print(references)
            bleu_, _, _, _, _, _ = bleu.compute_bleu(references, translations)
            bleu_score += 100.0 * bleu_
        bleu_score = bleu_score * 1.0 / (idx+1)
        return bleu_score

    def get_keys(d, value):
        return [k for k,v in d.items() if v == value]            

    # def get_evaluation_from_batches(self, sess, batches):
    #     e = sum(self.get_evaluation(sess, batch) for batch in batches)
    #     return e
        