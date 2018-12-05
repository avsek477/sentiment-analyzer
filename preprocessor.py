import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

REPLACE_WITH_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile(r"<.*?>|(\-)|(\/)")
le = LabelEncoder()

def stemma(review):
    stemmer = SnowballStemmer("english")
    stemmedData = []
    word_tokens = word_tokenize(review)
    for word in word_tokens:
        stemmedData.append(stemmer.stem(word))
    return " ".join(stemmedData)

def preprocess_data(data):
    data['label'] = le.fit_transform(data['label'])
    data['review'] = [REPLACE_WITH_NO_SPACE.sub("", review.lower()) for review in data['review']]
    data['review'] = [REPLACE_WITH_SPACE.sub(" ", review) for review in data['review']]
    data['review'] = [stemma(review) for review in data['review']]
    return data