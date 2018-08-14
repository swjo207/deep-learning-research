import math
import sys

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

pairs = [
	['The frog catched a big fly', 'Frog catched a big fly'],
	['It swallowed the fly', 'It eated the fly'],
]
refs = [p[0].lower().split() for p in pairs]
candis = [p[1].lower().split() for p in pairs]
print(get_bleu(refs, candis))

#sys.path.append('/d/aep/test_tensorflow/test_seq2seq/t/nmt/scripts'); import bleu
#refs = [[p[0].lower().split()] for p in pairs]
#print(bleu.compute_bleu(refs, candis)[0])
