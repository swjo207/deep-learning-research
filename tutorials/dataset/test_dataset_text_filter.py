import tensorflow as tf

def print_dataset(ds):
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

paths = []
for i in range(2):
	path = 'test_filter%d.txt' % i; paths.append(path)
	with open(path, 'w') as f:
		f.write(
"""This will be removed
%d, 2
%d, 4
# This will be removed
%d, 6
""" % (i*10, i*10+1, i*10+2))

with tf.Session() as sess:
	print(tf.decode_csv('1,2', [[], []]))

	def parse(path):
		d = tf.data.TextLineDataset(path).skip(1).filter(
			lambda l: tf.not_equal(tf.substr(l, 0, 1), '#'))
		return d.map(lambda l: tf.decode_csv(l, [[], []]))
	ds = tf.data.Dataset.from_tensor_slices(paths).flat_map(parse)
					
	print_dataset(ds)

