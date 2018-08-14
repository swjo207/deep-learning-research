import os
import random
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

FEATURE_COLS = [
	'symbolings', 'normalized-losses', 'make', 'fuel_type', 'aspirations',
	'n_doors', 'body_style', 'drive_wheels', 'engine_loc', 'wheel_base',
	'length', 'width', 'height', 'curb_weight', 'engine_type',
	'n_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',
	'compression-ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg',
	'price']
BATCH_SIZE = 100
PRICE_NORM_FACTOR = 1000.

def create_data_files():
	# https://archive.ics.uci.edu/ml/datasets/automobile
	# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
	if os.path.exists('imports-85.data.slow2.train'): return
	with open('imports-85.data', 'r') as sf, \
		open('imports-85.data.slow2.train', 'w') as train_f, \
		open('imports-85.data.slow2.test', 'w') as test_f:
		lines = [l for l in sf.readlines() if not '?' in l]
		random.shuffle(lines)
		n_tests = int(len(lines) * 0.3)
		train_f.write(''.join(lines[0:n_tests]))
		test_f.write(''.join(lines[n_tests:]))

def create_ds(path):
	ds = tf.data.TextLineDataset(path)
	def parse(line):
		*feature, label = tf.decode_csv(line, [
			[0], [0.], [''], [''], [''],		[''], [''], [''], [''], [0.0],
			[0.], [0.], [0.], [0.], [''],		[''], [0.], [''], [0.], [0.],
			[0.], [0.], [0.], [0.], [0.],		[0.]])
		label /= PRICE_NORM_FACTOR
		return (dict(zip(FEATURE_COLS, feature)), label)
	ds = ds.map(parse)
	return ds.shuffle(1000)

def create_train_ds(): return create_ds('imports-85.data.slow2.train').repeat().batch(BATCH_SIZE)
def create_test_ds(): return create_ds('imports-85.data.slow2.test').batch(BATCH_SIZE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

create_data_files()
os.system('rm -rf model/*')
es = tf.estimator.DNNRegressor(
	[16, 16], # two hidden layers
	[tf.feature_column.numeric_column('curb_weight'),
		tf.feature_column.numeric_column('highway_mpg'),
		tf.feature_column.indicator_column(
			tf.feature_column.categorical_column_with_vocabulary_list('body_style', v_body_style)),
		tf.feature_column.embedding_column(
			tf.feature_column.categorical_column_with_vocabulary_list('make', v_make), 3),
	],
	model_dir = 'model')
print('*train')
es.train(create_train_ds, steps = 2000, hooks = [TrainHook()])
print('*evaluate')
r = es.evaluate(create_test_ds)
print(', '.join(['%s:%s' % (k, r[k]) for k in sorted(r.keys())]))
print('	rms error:%s' % (PRICE_NORM_FACTOR * r['average_loss'] ** 0.5))


