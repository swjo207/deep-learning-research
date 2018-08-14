import os
import numpy
import pandas
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

v_make = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge',
	'honda', 'isuzu', 'jaguar', 'mazda', 'mercedes-benz',
	'mercury', 'mitsubishi', 'nissan', 'peugot', 'plymouth',
	'porsche', 'renault', 'saab', 'subaru', 'toyota',
	'volkswagen', 'volvo']
v_fuel_type = ['diesel', 'gas']
v_aspirations = ['std', 'turbo']
v_n_doors = ['four', 'two']
v_body_style = ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible']
v_drive_wheels = ['4wd', 'fwd', 'rwd']
v_engine_loc = ['front', 'rear']
v_engine_type = ['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor']
v_n_cylinders = ['eight', 'five', 'four', 'six', 'three', 'twelve', 'two']
v_fuel_system = ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi']

COLS = [
	'symbolings', 'normalized-losses', 'make', 'fuel_type', 'aspirations',
	'n_doors', 'body_style', 'drive_wheels', 'engine_loc', 'wheel_base',
	'length', 'width', 'height', 'curb_weight', 'engine_type',
	'n_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',
	'compression-ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg',
	'price']
TYPE_BY_COL = {c: str if 'v_' + c in globals() else numpy.float32  for c in COLS}
feature_cols = [
	tf.feature_column.numeric_column('curb_weight'),
	tf.feature_column.numeric_column('highway_mpg'),
	tf.feature_column.indicator_column(
		tf.feature_column.categorical_column_with_vocabulary_list('body_style', v_body_style)),
	tf.feature_column.embedding_column(
		tf.feature_column.categorical_column_with_vocabulary_list('make', v_make), 3),
]
BATCH_SIZE = 100
PRICE_NORM_FACTOR = 1000.

def create_data():
	# https://archive.ics.uci.edu/ml/datasets/automobile
	# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
	f = pandas.read_csv('imports-85.data', names=COLS,
		dtype=TYPE_BY_COL, na_values="?")
	f = f.dropna()
	train_f = f.sample(frac = 0.7)
	test_f = f.drop(train_f.index)
	return train_f, test_f

def create_ds(frame):
	labels = numpy.array(frame.pop('price')) / PRICE_NORM_FACTOR
	features = {k: numpy.array(v) for k, v in frame.items()}
	return tf.data.Dataset.from_tensor_slices((features, labels))
def create_train_ds(): return create_ds(train_frame).repeat().batch(BATCH_SIZE)
def create_test_ds(): return create_ds(test_frame).batch(BATCH_SIZE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf model/*')
train_frame, test_frame = create_data()

def create_model(features, labels, mode):
	inputs = tf.feature_column.input_layer(features, feature_cols)
	for n in [16, 16]: # two hidden layers
		inputs = tf.layers.dense(inputs, n, activation = tf.nn.relu)
	logits = tf.layers.dense(inputs, 1) # 1: predicting a scalar
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'predictions': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions = predictions)
	else:
		logits_vec = tf.squeeze(logits, 1)
		average_loss = tf.losses.mean_squared_error(labels, logits_vec)
		loss = tf.to_float(tf.shape(labels)[0]) * average_loss		
		if mode == tf.estimator.ModeKeys.EVAL:
			average_loss_and_op = tf.metrics.mean_squared_error(labels, logits_vec)
			metric_ops = {'average_loss': average_loss_and_op}
			return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metric_ops)
		else:
			optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1)
			train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

es = tf.estimator.Estimator(create_model, model_dir = 'model2')
print('*train')
es.train(create_train_ds, steps = 2000, hooks = [TrainHook()])
print('*evaluate')
r = es.evaluate(create_test_ds)
print(', '.join(['%s:%s' % (k, r[k]) for k in sorted(r.keys())]))
print('	rms error:%s' % (PRICE_NORM_FACTOR * (r['average_loss'] ** 0.5)))
print('*predict')
for f in [
		[2548, 27, 'convertible', 'alfa-romero'], # 13495
		[2507, 25, 'sedan', 'audi'], # 15250
	]:
	def feature_col_name(c):
		return c.categorical_column.key if hasattr(c, 'categorical_column') else c.key
	fd = dict(zip([feature_col_name(c) for c in feature_cols], [[v] for v in f]))
	r = es.predict(lambda: tf.data.Dataset.from_tensor_slices(fd).batch(1))
	for e in r:
		print('price:%s' % (e['predictions'][0] * PRICE_NORM_FACTOR))
