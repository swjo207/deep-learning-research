import collections
import itertools
import math
import random
import time
import os
import _pickle as cPickle
import numpy
import tensorflow as tf

class TrainHook(tf.train.SessionRunHook):
	def begin(self):
		self.i = 0
		self.t = time.time()
	def before_run(self, ctx):
		# See codes around 'GraphKeys.LOSSES' in Estimator._train_model()
		if self.i % 2000 == 0:
			return tf.train.SessionRunArgs({'loss': tf.losses.get_losses()[-1]})
		return None
	def after_run(self, ctx, values):
		if values.results:
			new_t = time.time()
			print('dt:%.1f, step:%d, %s' % (new_t - self.t, self.i, values.results))
			self.t = new_t
		self.i += 1

VOC_SIZE = 50000
N_NEG_SAMPLES = 64
feature_cols = [
	tf.feature_column.embedding_column(
		tf.feature_column.categorical_column_with_identity('word', VOC_SIZE), 128),
]
BATCH_SIZE = 128

def create_data():
	pickle_path = 'text8.pickle'
	if os.path.exists(pickle_path):
		with open(pickle_path, 'rb+') as f:
			r = cPickle.load(f)
	else:
		# http://mattmahoney.net/dc/text8.zip
		with open('text8', 'rt') as f:
			words = f.read().split()
		counts = {}
		for w in words:
			counts[w] = counts.get(w, 0) + 1
		counts = sorted(counts.items(), key = lambda kv: kv[1], reverse = True)[:(VOC_SIZE-1)]
		word_to_id = {'_dropped_': 0}
		voc_words = ['_dropped_']
		for w, _ in counts:
			word_to_id[w] = len(voc_words)
			voc_words.append(w)
		data_word_ids = [word_to_id.get(w, 0) for w in words]
		r = word_to_id, voc_words, data_word_ids
		with open(pickle_path, 'wb+') as f:
			cPickle.dump(r, f)
	return r

"""
def create_train_ds():
	c = 1 # context size
	all_neigh_offsets = [i for i in range(2*c+1) if i != c] # offset of all neighbors around a center word
	n_neigh = 2 # number of neighbors per a center word, an input word
	n_words = len(data_word_ids) - 2*c
	def gen():
		i = 0
		features = {'word': 0}
		while True:
			features['word'] = data_word_ids[i + c]
			for o in random.sample(all_neigh_offsets, n_neigh):
				label = data_word_ids[i + o]
				yield (features, label)
			i = (i + 1) % n_words
	return tf.data.Dataset.from_generator(gen, ({'word': tf.int32}, tf.int32)).batch(BATCH_SIZE)
"""
def create_train_ds():
	c = 1 # context size
	all_neigh_offsets = [i for i in range(2*c+1) if i != c] # offset of all neighbors around a center word
	n_neigh = 2 # number of neighbors per a center word, an input word
	n_words = len(data_word_ids) - 2*c
	def gen():
		i = 0
		feature_word = numpy.zeros((BATCH_SIZE,), 'int32')
		label = numpy.zeros((BATCH_SIZE,), 'int32')
		while True:
			for j in range(BATCH_SIZE // n_neigh):
				s = j * n_neigh
				feature_word[s:s + n_neigh] = data_word_ids[i + c]
				label[s:s + n_neigh] = [data_word_ids[i + o] for o in random.sample(all_neigh_offsets, n_neigh)]
				i = (i + 1) % n_words
			yield ({'word': feature_word}, label)
	return tf.data.Dataset.from_generator(gen, ({'word': tf.int32}, tf.int32))

word_to_id, voc_words, data_word_ids = create_data()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf model/*')

def create_model(features, labels, mode):
	inputs = tf.feature_column.input_layer(features, feature_cols)
	with tf.name_scope('nce'):
		dim_embed = int(inputs.shape[1])
		# output vector representations
		weights = tf.Variable(tf.truncated_normal([VOC_SIZE, dim_embed], stddev = 1/math.sqrt(dim_embed)))
		biases = tf.Variable(tf.zeros([VOC_SIZE]))
	labels = tf.reshape(labels, [-1, 1])
	loss = tf.reduce_mean(tf.nn.nce_loss(weights, biases, labels, inputs,
		N_NEG_SAMPLES, VOC_SIZE))
	#optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1)
	train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

es = tf.estimator.Estimator(create_model, model_dir = 'model')
print('*train')
es.train(create_train_ds, steps = 100001, hooks = [TrainHook()])



def plot_with_labels(low_dim_embs, labels, filename):
	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label, xy = (x, y), xytext = (5, 2),
			textcoords = 'offset points', ha = 'right', va = 'bottom')
	plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000, method = 'exact')
plot_only = 500
word_vecs = es.get_variable_value('input_layer/word_embedding/embedding_weights')
normalized_word_vecs = word_vecs / numpy.linalg.norm(word_vecs, axis = 1)[:, None]
low_dim_embs = tsne.fit_transform(normalized_word_vecs[:plot_only, :])
labels = [voc_words[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, 'tsne_old.png')
