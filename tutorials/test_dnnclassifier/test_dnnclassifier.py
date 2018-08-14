import os
import tensorflow as tf

class TrainHook(tf.train.SessionRunHook):
	def begin(self):
		self.i = 0
	def before_run(self, ctx):
		# See codes around 'GraphKeys.LOSSES' in Estimator._train_model()
		if self.i % 100 == 0: return tf.train.SessionRunArgs(
			{'loss': tf.losses.get_losses()[-1]})
		return None
	def after_run(self, ctx, values):
		if values.results:
			print(self.i, values.results)
		self.i += 1

FEATURE_COLS = ['sl', 'sw', 'pl', 'pw']
BATCH_SIZE = 100

def create_ds(path):
	# http://download.tensorflow.org/data/iris_training.csv'
	# http://download.tensorflow.org/data/iris_test.csv'
	ds = tf.data.TextLineDataset(path)
	def parse(line):
		*feature, label = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
		return (dict(zip(FEATURE_COLS, feature)), label)
	ds = ds.skip(1).map(parse)
	return ds

def create_train_ds():
	ds = create_ds('iris_training.csv')
	ds = ds.shuffle(1000).repeat()
	return ds.batch(BATCH_SIZE)

def create_test_ds():
	ds = create_ds('iris_test.csv')
	return ds.batch(BATCH_SIZE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf model/*')
es = tf.estimator.DNNClassifier(
	[8, 8], # two hidden layers
	[tf.feature_column.numeric_column(k) for k in FEATURE_COLS],
	n_classes = 3,
	model_dir = 'model')
print('*train')
es.train(create_train_ds, steps = 1000, hooks = [TrainHook()])
print('*evaluate')
r = es.evaluate(create_test_ds)
print(', '.join(['%s:%s' % (k, r[k]) for k in sorted(r.keys())]))
print('*predict')
for f in [[5.1, 3.3, 1.7, 0.5],
	[5.9, 3.0, 4.2, 1.5],
	[6.9, 3.1, 5.4, 2.1]]:
	fd = dict(zip(FEATURE_COLS, [[v] for v in f]))
	r = es.predict(lambda: tf.data.Dataset.from_tensor_slices(fd).batch(1))
	for e in r:
		sp = e['class_ids'][0]
		print(sp, e['probabilities'][sp])
