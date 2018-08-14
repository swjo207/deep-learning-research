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

os.system('rm -rf model2/*')

def create_model(features, labels, mode):
	inputs = tf.feature_column.input_layer(features,
		[tf.feature_column.numeric_column(k) for k in FEATURE_COLS])
	for n in [8, 8]: # two hidden layers
		inputs = tf.layers.dense(inputs, n, activation = tf.nn.relu)
	logits = tf.layers.dense(inputs, 3) # 3: number of classes
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
			optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1)
			train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

es = tf.estimator.Estimator(create_model, model_dir = 'model2')
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
