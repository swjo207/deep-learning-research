import os
import math
import time
import functools
print = functools.partial(print, flush=True)
import _pickle as cPickle
import numpy
import tensorflow as tf

class TrainHook(tf.train.SessionRunHook):
	def begin(self):
		self.i = 0
		self.t = time.time()
		self.stats = None
	def before_run(self, ctx):
		# See codes around 'GraphKeys.LOSSES' in Estimator._train_model()
		if self.i % 500 == 0:
			return tf.train.SessionRunArgs({'global_step': tf.train.get_global_step(),
				'loss': tf.losses.get_losses()[-1]})
		return None
	def after_run(self, ctx, values):
		if values.results:
			new_t = time.time()
			self.stats = values.results
			print('  dt:%.1f, %s' % (new_t - self.t, self.stats))
			self.t = new_t
		self.i += 1

train_hook = TrainHook()

def get_bleu(references, candidates):
	max_n = 4
	sums_clipped_counts_n_gram = [0 for _ in range(max_n)]
	sums_counts_n_gram = [0 for _ in range(max_n)]
	sum_ref_lens = sum_candi_lens = 0
	for ref, candi in zip(references, candidates):
		sum_ref_lens += len(ref)
		candi_len = len(candi); sum_candi_lens += candi_len
		def count_n_grams(sen):
			sen_len = len(sen)
			counts = dict()
			for n in range(1, 1 + max_n):
				for j in range(sen_len - n + 1):
					gram = tuple(sen[j:j+n])
					counts[gram] = counts.get(gram, 0) + 1
			return counts
		counts_ref = count_n_grams(ref)
		counts_candi = count_n_grams(candi)
		for gram, c_candi in counts_candi.items():
			c_ref = counts_ref.get(gram, 0)
			if c_ref > 0:
				sums_clipped_counts_n_gram[len(gram)-1] += min(c_candi, c_ref)
		for i in range(max_n):
			c = candi_len - i
			if c > 0: sums_counts_n_gram[i] += c
	sum_log_p = 0
	for i in range(max_n):
		d = sums_counts_n_gram[i]
		if d == 0: return 0
		p = float(sums_clipped_counts_n_gram[i]) / d
		if p == 0: return 0
		sum_log_p += math.log(p)
	bleu = math.exp((1.0/max_n * sum_log_p))
	if sum_candi_lens < sum_ref_lens:
		bleu *= math.exp(1 - float(sum_ref_lens) / sum_candi_lens)
	return bleu

# indexes of special words
DROPPED = 0
SOS = 1
EOS = 2

# hyper parameters
#DIM_WORD_EMBED = 512
DIM_WORD_EMBED = 128
MAX_SEN_SIZE = 50
SEN_SIZE_STEP = 10
BATCH_SIZE = 128
INITIALIZER_MAX = 0.1 # If this is 0.5/DIM_WORD_EMBED, the BLEU will be about 1/4 at 10 epoch.
MAX_GRAD_NORM = 5.0
LEARNING_RATE = 1
ENCODER_NUM_SUBCELL = 2
DECODER_NUM_SUBCELL = 2

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
			with open(path, 'rt', 'utf8') as f:
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
		with open(pickle_path.replace('.pickle', '.count.txt'), 'wt+', 'utf8') as f:
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
][0]
voc_lang1, train_lang1_data, test_lang1_data = create_data(dir, lang1, VOC_SIZE1, train_prefix + '.' + lang1, test_prefix + '.' + lang1)
voc_lang2, train_lang2_data, test_lang2_data = create_data(dir, lang2, VOC_SIZE2, train_prefix + '.' + lang2, test_prefix + '.' + lang2)

def create_train_ds():
	buf_size = BATCH_SIZE * 1000
	ranges1, words1 = train_lang1_data; ranges2, words2 = train_lang2_data
	ranges1 = ranges1.astype('int32'); ranges2 = ranges2.astype('int32')
	words1 = tf.constant(words1, tf.int32); words2 = tf.constant(words2, tf.int32)
	ds = tf.data.Dataset.from_tensor_slices((ranges1, ranges2)).\
		filter(lambda r1, r2: tf.logical_and(r1[1] > 0, r2[1] > 0)).\
		shuffle(buf_size)
	def conv(r1, r2):
		n1 = tf.minimum(r1[1], MAX_SEN_SIZE)
		n2 = tf.minimum(r2[1], MAX_SEN_SIZE)
		label_words = words2[r2[0]:r2[0] + n2]
		return ((words1[r1[0]:r1[0] + n1], n1),
			(tf.concat(([SOS], label_words), 0), tf.concat((label_words, [EOS]), 0), n2 + 1))
	ds = ds.map(conv).prefetch(buf_size)
	return ds.apply(tf.contrib.data.group_by_window(
		lambda features, labels: tf.to_int64(tf.maximum(features[1], labels[2]) // SEN_SIZE_STEP),
		lambda key, ds_: ds_.padded_batch(BATCH_SIZE,
			((tf.TensorShape([None]), tf.TensorShape([])),
				(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))),
			((EOS, 0), (EOS, EOS, 0))),
		window_size = BATCH_SIZE))

def create_test_ds():
	ranges1, words1 = test_lang1_data
	ranges1 = ranges1.astype('int32')
	words1 = tf.constant(words1, tf.int32)
	ds = tf.data.Dataset.from_tensor_slices(ranges1)
	def conv(r1):
		n1 = r1[1]
		return ((words1[r1[0]:r1[0] + n1], n1),)
	ds = ds.map(conv)
	return ds.padded_batch(BATCH_SIZE,
		((tf.TensorShape([None]), tf.TensorShape([])),),
		((EOS, 0),))

def create_test_bleu_refs():
	ranges2, words2 = test_lang2_data
	refs = []
	for r2 in ranges2:
		n2 = r2[1]
		refs.append(words2[r2[0]:r2[0] + n2])
	return refs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.INFO)

os.system('rm -rf %s/model.simple/*' % dir)

def create_model(features, labels, mode):
	initializer = tf.random_uniform_initializer(-INITIALIZER_MAX, INITIALIZER_MAX)
	tf.get_variable_scope().set_initializer(initializer)

	features_sen, features_len = features
	features_sen_shape = tf.shape(features_sen)
	batch_size = features_sen_shape[0]; features_max_len = features_sen_shape[1]

	word_vecs1 = tf.get_variable('word_vecs1', [VOC_SIZE1, DIM_WORD_EMBED], tf.float32)
	features_sen_embed = tf.nn.embedding_lookup(word_vecs1, features_sen)

	word_vecs2 = tf.get_variable('word_vecs2', [VOC_SIZE2, DIM_WORD_EMBED], tf.float32)

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
		create_rnn_cell(ENCODER_NUM_SUBCELL), features_sen_embed, dtype = tf.float32,
		sequence_length = features_len, swap_memory = True)

	decoder_cell = create_rnn_cell(DECODER_NUM_SUBCELL)
	out_dense = tf.layers.Dense(VOC_SIZE2, use_bias = False)

	if mode == tf.estimator.ModeKeys.TRAIN:
		labels_sen_with_sos, labels_sen_with_eos, labels_len = labels
		labels_sen_embed = tf.nn.embedding_lookup(word_vecs2, labels_sen_with_sos)

		helper = tf.contrib.seq2seq.TrainingHelper(labels_sen_embed, labels_len)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state)
		with tf.variable_scope('decoder') as scope:
			decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory = True, scope = scope)
			logits = out_dense(decoder_out.rnn_output)
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_sen_with_eos, logits = logits)
		losses_weights = tf.sequence_mask(labels_len, tf.shape(labels_sen_with_eos)[1], dtype = logits.dtype)
		loss = tf.reduce_sum(losses * losses_weights) / tf.to_float(batch_size)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
		#train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
		grads, vars = zip(*optimizer.compute_gradients(loss))
		grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
		train_op = optimizer.apply_gradients(zip(grads, vars), global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)
	else:
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_vecs2,
			tf.fill([batch_size], SOS), EOS)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
			output_layer = out_dense)
		with tf.variable_scope('decoder') as scope:
			decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
				maximum_iterations = features_max_len * 2, swap_memory = True, scope = scope)
		preds = {'predicted_sen': decoder_out.sample_id}
		return tf.estimator.EstimatorSpec(mode, predictions = preds)

cfg = tf.estimator.RunConfig(save_checkpoints_steps = 100)
args = {}
#args['config'] = cfg
es = tf.estimator.Estimator(create_model, model_dir = dir + '/model.simple', **args)
test_refs = create_test_bleu_refs()

t0 = time.time()
for _ in range(30):
	print('  begin to train 1 epoch')
	es.train(create_train_ds, hooks = [train_hook])
	train_stats = train_hook.stats
	print('  begin to test')
	test_candis = []
	for preds in es.predict(create_test_ds, yield_single_examples = False):
		for sen in preds['predicted_sen']:
			if EOS in sen:
				sen = tuple(sen)
				sen = sen[:sen.index(EOS)]
			test_candis.append(sen)
	train_stats['bleu'] = bleu = get_bleu(test_refs, test_candis) * 100
	print('t:%.1f, %s' % (time.time() - t0, train_stats))
	writer = tf.summary.FileWriterCache.get(es.model_dir + '/test')
	writer.add_summary(tf.Summary(value = [
			tf.Summary.Value(tag = 'bleu', simple_value = bleu)]),
		train_stats['global_step'])
	writer.flush()
