import tensorflow as tf

path = 'test.txt'
with open(path, 'w') as f:
	f.write("a\nb\nc")

with tf.Session() as sess:
	ds = tf.data.TextLineDataset(path)
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = a.eval()
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass
