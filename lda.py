import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import pickle
import gensim
import pyLDAvis
import pyLDAvis.gensim
import gensim.corpora as corpora


class PreProcess:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemma = WordNetLemmatizer()

    # remove non-alpha characters 
    def translate(self, text):
        output = []
        for word in word_tokenize(text):
            temp_word = ''
            for w in word:
                if not w.isalpha(): 
                    continue
                temp_word += w
            if temp_word: 
                output.append(temp_word)
        return ' '.join(output)

    def remove_stopwords(self, text):
        tag_re = re.compile(r'<[^>]+>')
        text = tag_re.sub('', text)
        # tbl = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        # text = text.translate(tbl)
        text = self.translate(text)
        text = ' '.join([w for w in word_tokenize(text) if not w in self.stop_words])
        return text

    def lemmatize(self, text):
        text = word_tokenize(text)
        text = [self.lemma.lemmatize(self.lemma.lemmatize(w), pos='v') for w in text]
        return text


class LDA:
    def __init__(self, all_data=None):
        if all_data is None:
            self.dict = None
            self.corpus = None
        else:
            self.dict = corpora.dictionary.Dictionary(all_data)
            self.corpus = [self.dict.doc2bow(text) for text in all_data]
        self.model = None

    def build_model(self, n_topics):
        self.model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                num_topics=n_topics)

    # needs a single dimensional list of tokens
    def get_distribution(self, corpus):
        if self.model is None:
            print 'You need to build the model first!'
            return
        corpus = self.dict.doc2bow(corpus)
        vector = self.model[corpus]
        return vector

    def print_model(self):
        if self.model is None:
            print 'You need to build the model first!'
            return
        topics = self.model.print_topics()
        for topic in topics:
            text = ''
            temp = topic[1]
            pairs = re.findall(r'''(0\.[^\*]+)\*"([^"]+)"''', temp)
            for pair in pairs:
                text += pair[0]+'*'+self.dict[int(pair[1])] + ' + '
            print text[:-3]

    def print_topic(self, topic_id):
        topic = self.model.show_topic(topic_id)
        text = ''
        pairs = re.findall(r'''(0\.[^\*]+)\*"([^"]+)"''', topic)
        for pair in pairs:
            text += pair[0]+'*'+self.dict[int(pair[1])] + ' + '
        print text[:-3]

    def ntopics(self):
        return self.model.get_topics().shape[0] 

    def save_model(self, filename):
        if self.model is None:
            print 'You need to build the model first!'
            return
        self.model.save(filename)
        self.dict.save_as_text(filename+'_dict')

    def load_model(self, filename):
        self.model = gensim.models.ldamodel.LdaModel.load(filename)
        self.dict = corpora.dictionary.Dictionary.load_from_text(filename+'_dict') 

'''
joined = pd.read_pickle('movies_joined')
processor = PreProcess()
all_text = [] 
for i in range(joined.shape[0]):
    text = ''
    if joined.iloc[i]['Title'] is not None:
        text += ' '+joined.iloc[i]['Title'].lower()+' '
    if joined.iloc[i]['Body'] is not None:
        text += joined.iloc[i]['Body'].lower()+' '
    if joined.iloc[i]['Tags'] is not None:
        tags = ' '.join(joined.iloc[i]['Tags'][1:-1].split('><'))
        text += tags.lower()
    if text:
        text = processor.lemmatize(processor.remove_stopwords(text))
        all_text.append(text)

# --------------------------------------------------
# Please pay attention to these details
with open('all_posts.pkl', 'wb') as f:
    pickle.dump(all_text, f)

# ---------------------------------------------------
model = LDA(all_text)
model.build_model(n_topics=50)
model.print_model()
model.save_model('movies')
'''

if __name__ == '__main__':
    joined = pd.read_pickle('movies_joined')
    processor = PreProcess()
    all_text = [] 
    for i in range(joined.shape[0]):
        text = ''
        if joined.iloc[i]['Title'] is not None:
            text += ' '+joined.iloc[i]['Title'].lower()+' '
        if joined.iloc[i]['Body'] is not None:
            text += joined.iloc[i]['Body'].lower()+' '
        if joined.iloc[i]['Tags'] is not None:
            tags = ' '.join(joined.iloc[i]['Tags'][1:-1].split('><'))
            text += tags.lower()
        if text:
            text = processor.lemmatize(processor.remove_stopwords(text))
            all_text.append(text)

    '''
    with open('all_movies.pkl', 'wb') as f:
       pickle.dump(all_text, f)
    
    f = open('all_movies.pkl', 'rb')
    all_text = pickle.load(f)
    ''' 
        
    model = LDA(all_text)
    model.build_model(n_topics=20)
    # model.print_model()
    model.save_model('movies_20')
	
    # Uncomment to create visualization of topics
    '''
    corpus = [model.dict.doc2bow(text) for text in all_text]
    pyLDAvis.enable_notebook()
    vis=pyLDAvis.gensim.prepare(model.model, corpus, model.dict)
    '''
