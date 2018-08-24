from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import string
import re

# 텍스트 데이터 로딩
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt', encoding='utf-8')
text = file.read()
file.close()

# 문장으로 분리
sentences = sent_tokenize(text)
print(sentences[0])

# 단어 단위로 분리
tokens = word_tokenize(text)
print(tokens[:100])

# 알파벳이 아닌 토큰 제거
words = [word for word in tokens if word.isalpha()]
print(words[:100])

# 정규표현을 이용하여 불용어 제거
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# 구두점 제거
stripped = [re_punc.sub('', w) for w in tokens]
# 알파벳이 아닌 기타 문자 제거
words = [word for word in stripped if word.isalpha()]
# 불용어 제거
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])

# 어근(stemming) 분리
# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])