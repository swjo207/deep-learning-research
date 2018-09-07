# -*- coding: utf-8 -*- 
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import Tokenizer

"""
1. Keras text_to_word_sequence()  
"""
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)

"""
2. Keras 단어 사전   
"""
# estimate the size of the vocabulary
words = set(result)
vocab_size = len(words)
print(vocab_size)
"""
3. Keras 원-핫 인코딩   
"""
# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)

"""
4. Keras 해시 인코딩   
"""
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)

"""
5. Keras 토큰 분리(Tokenization)    
"""
# define 5 documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_docs)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)