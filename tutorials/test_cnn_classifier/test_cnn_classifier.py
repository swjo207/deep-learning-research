import os
import time
import _pickle as cPickle
import numpy
import tensorflow as tf

class TrainHook(tf.train.SessionRunHook):
	def begin(self):
		self.i = 0
		self.t = time.time()
	def before_run(self, ctx):
		# See codes around 'GraphKeys.LOSSES' in Estimator._train_model()
		if self.i % 600 == 0:
			return tf.train.SessionRunArgs({'loss': tf.losses.get_losses()[-1]})
		return None
	def after_run(self, ctx, values):
		if values.results:
			new_t = time.time()
			print('dt:%.1f, step:%d, %s' % (new_t - self.t, self.i, values.results))
			self.t = new_t
		self.i += 1

IMG_W, IMG_H = 28, 28
BATCH_SIZE = 100

def create_data():
	pickle_path = 'data.pickle'
	if os.path.exists(pickle_path):
		with open(pickle_path, 'rb+') as f:
			r = cPickle.load(f)
	else:
		# https://storage.googleapis.com/cvdf-datasets/mnist
		# train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz
		# t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
		def parse(imgs_path, labels_path):
			num_imgs = (os.path.getsize(imgs_path) - 16) // (IMG_W * IMG_H)
			with open(imgs_path, 'rb') as imgs_f, open(labels_path, 'rb') as labels_f:
				imgs_f.read(16); labels_f.read(8)
				bytes = imgs_f.read(IMG_W * IMG_H * num_imgs)
				imgs = numpy.frombuffer(bytes, 'uint8').astype('float32') / 255
				imgs = imgs.reshape(num_imgs, 1, IMG_H, IMG_W)
				bytes = labels_f.read(num_imgs)
				labels = numpy.frombuffer(bytes, 'uint8').astype('int32')
				return imgs, labels
		train_data = parse('t2/mnist_data/train-images-idx3-ubyte', 't2/mnist_data/train-labels-idx1-ubyte')
		test_data = parse('t2/mnist_data/t10k-images-idx3-ubyte', 't2/mnist_data/t10k-labels-idx1-ubyte')
		r = train_data, test_data
		with open(pickle_path, 'wb+') as f:
			cPickle.dump(r, f)
	return r

train_data, test_data = create_data()

"""
def create_ds(imgs, labels):
	return tf.data.Dataset.from_tensor_slices((imgs, labels))
def create_train_ds(): return create_ds(*train_data).shuffle(60000).repeat().batch(BATCH_SIZE)
def create_test_ds(): return create_ds(*test_data).batch(BATCH_SIZE)
"""
# The from_tensor_slices() is too slow for the train data. (It takes about 20 seconds in my machine.)
# But from_tensor_slices() is fast for the test data when evaluating.
def create_train_ds():
	def gen():
		imgs, labels = train_data
		n_imgs = imgs.shape[0]
		i = n_imgs - BATCH_SIZE
		while True:
			new_i = i + BATCH_SIZE
			if new_i >= n_imgs:
				ids = numpy.random.permutation(n_imgs)
				imgs, labels = imgs[ids], labels[ids]
				new_i = 0
			i = new_i
			yield (imgs[i:i+BATCH_SIZE, ...], labels[i:i+BATCH_SIZE])
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32),
		(tf.TensorShape([BATCH_SIZE, 1, IMG_H, IMG_W]), tf.TensorShape([BATCH_SIZE])))
def create_test_ds():
	return tf.data.Dataset.from_tensor_slices(test_data).batch(BATCH_SIZE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf model/*')

def create_model(features, labels, mode):
	filter_args = dict(padding = 'same', data_format = 'channels_first')
	y = tf.layers.conv2d(features, 32, 5, activation = tf.nn.relu, **filter_args) # 5x5 kernel, depth 32
	y = tf.layers.max_pooling2d(y, (2, 2), (2, 2), **filter_args)

	y = tf.layers.conv2d(y, 64, 5, activation = tf.nn.relu, **filter_args)
	y = tf.layers.max_pooling2d(y, (2, 2), (2, 2), **filter_args)

	y = tf.layers.flatten(y)
	y = tf.layers.dense(y, 1024, activation = tf.nn.relu)
	if mode == tf.estimator.ModeKeys.TRAIN: y = tf.layers.dropout(y, 0.4)
	logits = tf.layers.dense(y, 10)
	predicted_classes = tf.argmax(logits, 1)

	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predicted_classes[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions = predictions)
	else:
		# labels are class indexes.
		loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
		accuracy_and_op = tf.metrics.accuracy(labels = labels, predictions = predicted_classes,
			name='acc_op')
		tf.summary.scalar('accuracy', accuracy_and_op[1])

		if mode == tf.estimator.ModeKeys.EVAL:
			metric_ops = {'accuracy': accuracy_and_op}
			return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metric_ops)
		else:
			optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
			train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

es = tf.estimator.Estimator(create_model, model_dir = 'model')
print('*train')
n_train_samples = train_data[0].shape[0]
es.train(create_train_ds, steps = 40 * (n_train_samples // BATCH_SIZE), hooks = [TrainHook()])
print('*evaluate')
r = es.evaluate(create_test_ds)
print(', '.join(['%s:%s' % (k, r[k]) for k in sorted(r.keys())]))
