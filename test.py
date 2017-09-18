# dict1 = {'a':1,'b':2}
# dict2 = {v:k for k,v in dict1.items()}
# print(dict2)
# a = ["you","are","sb"]
# b = "".join(a)
# print(type(b))
import numpy as np
import tensorflow as tf

class Model(object):
	"""docstring for Model"""
	def __init__(self):
		self.isTrue = tf.placeholder('bool', shape=[], name='sb')

		if self.isTrue:
			print('sb1')
		else:
			print('sb2')

model = Model()

sess = tf.Session()

sess.run([], feed_dict={model.isTrue:False})
