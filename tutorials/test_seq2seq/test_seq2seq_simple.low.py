import os
import time
import _pickle as cPickle
import numpy
import tensorflow as tf

DROPPED = 0
SOS = 1
EOS = 2
#DIM_WORD_EMBED = 512
DIM_WORD_EMBED = 128
MAX_SEN_SIZE = 50
SEN_SIZE_STEP = 10
BATCH_SIZE = 128

def create_data(dir, lang, voc_size, train_src_file, test_src_file):
	pickle_path = dir + '/' + lang + '.pickle'
	if os.path.exists(pickle_path):
		with open(pickle_path, 'rb+') as f:
			r = cPickle.load(f)
	else:
		def parse(file):
			path = dir + '/' + file
			i = 0
			print('parsing ' + path)
			with open(path, 'rt') as f:
				while True:
					l = f.readline()
					if not l: break
					yield l.rstrip().split()
					i += 1
					if i % 500000 == 0: print('  %d lines' % i)
		counts = {}
		n_sens = 0; n_words = 0
		for words in parse(train_src_file):
			for w in words:
				counts[w] = counts.get(w, DROPPED) + 1
			n_sens += 1; n_words += len(words)
		voc_words = ['_dropped_', '_sos_', '_eos_']
		counts = sorted(counts.items(), key = lambda kv: kv[1], reverse = True)[:(voc_size-len(voc_words))]
		with open(pickle_path.replace('.pickle', '.count.txt'), 'wt+') as f:				
			f.write('\n'.join('%s:%d' % (k, v) for k,v in counts))
		word_to_id = {w: i for i, w in enumerate(voc_words)}
		for w, _ in counts:
			word_to_id[w] = len(voc_words)
			voc_words.append(w)
		r = [voc_words]
		for src_file in [train_src_file, test_src_file]:
			if len(r) == 2:
				n_sens = 0; n_words = 0
				for words in parse(src_file):
					n_sens += 1; n_words += len(words)
			sens_ranges = numpy.zeros((n_sens, 2), 'uint32')
			sens_words = numpy.zeros(n_words, 'uint16')
			i = 0; j = 0
			for words in parse(src_file):
				n = len(words)
				sens_ranges[i, :] = (j, n)
				sens_words[j:j + n] = tuple(word_to_id.get(w, DROPPED) for w in words)
				i += 1; j += n
			r.append((sens_ranges, sens_words))
		with open(pickle_path, 'wb+') as f:
			cPickle.dump(r, f)
	return r

# https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de
#	train.en, train.de, newstest2012.en, newstest2012.de
# https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi
#	train.en, train.vi, tst2013.en, tst2013.vi
dir, lang1, VOC_SIZE1, lang2, VOC_SIZE2, train_prefix, test_prefix = [
	['wmt14', 'en', 50000, 'de', 50000, 'train', 'newstest2012'],
	['iwslt15', 'en', 17191, 'vi', 7709, 'train', 'tst2013'],
][1]
voc_lang1, train_lang1_data, test_lang1_data = create_data(dir, lang1, VOC_SIZE1, train_prefix + '.' + lang1, test_prefix + '.' + lang1)
voc_lang2, train_lang2_data, test_lang2_data = create_data(dir, lang2, VOC_SIZE2, train_prefix + '.' + lang2, test_prefix + '.' + lang2)

def create_ds(lang1_data, lang2_data):
	def gen():
		ranges1, words1 = lang1_data
		ranges2, words2 = lang2_data
		#print(ranges1.shape[0], words1.shape[0])
		n_sens = ranges1.shape[0]
		i = n_sens; j = 0
		n_sen_groups = (MAX_SEN_SIZE + SEN_SIZE_STEP - 1 + 1) // SEN_SIZE_STEP
		sen_groups = [[] for _ in range(n_sen_groups)]
		while True:
			def find_data_in_group(group):
				n = len(group)
				max_n1 = max(e[1] for e in group); max_n2 = max(e[3] for e in group)
				features_sen = numpy.full((n, max_n1), EOS, 'int32')
				features_len = numpy.zeros(n, 'int32')
				labels_sen_with_sos = numpy.full((n, max_n2 + 1), EOS, 'int32')
				labels_sen_with_eos = numpy.full((n, max_n2 + 1), EOS, 'int32')
				labels_len = numpy.zeros(n, 'int32')
				for i, (s1, n1, s2, n2) in enumerate(group):
					sen1 = words1[s1:s1+n1]; sen2 = words2[s2:s2+n2]
					features_sen[i, :n1] = sen1
					features_len[i] = n1
					labels_sen_with_sos[i, 0] = SOS; labels_sen_with_sos[i, 1:1+n2] = sen2
					labels_sen_with_eos[i, :n2] = sen2;
					labels_len[i] = n2 + 1
				return ((features_sen, features_len), (labels_sen_with_sos, labels_sen_with_eos, labels_len))
			if i >= n_sens:
				for group in sen_groups:
					if group: yield find_data_in_group(group); group.clear()
				ids = numpy.random.permutation(n_sens)
				ranges1, ranges2 = ranges1[ids], ranges2[ids]
				i = 0
			s1, n1 = ranges1[i]; s2, n2 = ranges2[i]
			if n1 > 0 and n2 > 0:
				n1 = min(n1, MAX_SEN_SIZE)
				n2 = min(n2, MAX_SEN_SIZE)
				group = sen_groups[max(n1, n2+1) // SEN_SIZE_STEP]
				group.append([s1, n1, s2, n2])
				if len(group) >= BATCH_SIZE:
					yield find_data_in_group(group); group.clear()
			i += 1
	sen_shape = tf.TensorShape([None, None])
	len_shape = tf.TensorShape([None])
	return tf.data.Dataset.from_generator(gen,
		((tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32)),
		((sen_shape, len_shape), (sen_shape, sen_shape, len_shape)))

def create_train_ds(): return create_ds(train_lang1_data, train_lang2_data)
def create_test_ds(): return create_ds(test_lang1_data, test_lang2_data)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf %s/model/*' % dir)

def create_model(features, labels, mode):
	features_sen, features_len = features
	labels_sen_with_sos, labels_sen_with_eos, labels_len = labels
	word_vecs1 = tf.get_variable('word_vecs1', [VOC_SIZE1, DIM_WORD_EMBED], tf.float32)
	y1 = tf.nn.embedding_lookup(word_vecs1, features_sen)

	word_vecs2 = tf.get_variable('word_vecs2', [VOC_SIZE2, DIM_WORD_EMBED], tf.float32)
	y2 = tf.nn.embedding_lookup(word_vecs2, labels_sen_with_sos)

	def create_rnn_cell(num_subcells):
		cells = []
		for _ in range(num_subcells):
			cell = tf.nn.rnn_cell.BasicLSTMCell(DIM_WORD_EMBED)
			if mode == tf.estimator.ModeKeys.TRAIN:
				cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = 1.0 - 0.2)
			cells.append(cell)
		if num_subcells > 1: return tf.nn.rnn_cell.MultiRNNCell(cells)
		else: return cells[0]
	_, encoder_state = tf.nn.dynamic_rnn(
		create_rnn_cell(2), y1, dtype = tf.float32,
		sequence_length = features_len, swap_memory = True)

	decoder = tf.contrib.seq2seq.BasicDecoder(
		create_rnn_cell(2),
		tf.contrib.seq2seq.TrainingHelper(y2, labels_len),
		encoder_state)
	decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory = True)
	logits = tf.layers.dense(decoder_out.rnn_output, VOC_SIZE2, use_bias = False)
	if mode == 'predict':
		# TODO
		pass
	else:
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_sen_with_eos, logits = logits)
		losses_weights = tf.sequence_mask(labels_len, tf.shape(labels_sen_with_eos)[1], dtype = logits.dtype)
		loss = tf.reduce_sum(losses * losses_weights) / tf.to_float(BATCH_SIZE)
		if mode == 'eval':
			# TODO
			pass
		else:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1)
			#train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
			grads, vars = zip(*optimizer.compute_gradients(loss))
			grads, _ = tf.clip_by_global_norm(grads, 5.0)
			train_op = optimizer.apply_gradients(zip(grads, vars), global_step = tf.train.get_global_step())
			return loss, train_op

with tf.Session() as sess:
	ds = create_train_ds()
	features, labels = ds.make_one_shot_iterator().get_next()
	step = tf.train.create_global_step()
	d = 0.1
	initializer = tf.random_uniform_initializer(-d, d)
	tf.get_variable_scope().set_initializer(initializer)

	loss, train_op = create_model(features, labels, 'train')
	sess.run(tf.global_variables_initializer())
	t = time.time()
	while True:
		loss_v, _, step_v = sess.run((loss, train_op, step))
		if step_v % 100 == 0:
			new_t = time.time()
			print('dt:%.1f, step:%d, loss:%s' % (new_t - t, step_v, loss_v))
			t = new_t
		if step_v == 120000: break


