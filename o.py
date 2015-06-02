
import io
import numpy as np

from gensim.utils import tokenize
from gensim.models import Word2Vec, Phrases

#from nltk.corpus import stopwords
#eng_stops = stopwords.words('english')

vocab = [i.strip() for i in io.open('test-vocab.txt')]

def sent_vectorizer(sent, model):
    sent_vec = np.zeros(400)
    numw = 0
    for w in sent:
        if w not in vocab:
            continue
        try:
            sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    return sent_vec/numw

def cosine(s1, s2):
    cos_score = np.dot(s1, s2)
    if cos_score == np.NAN:
        return 9.9999e-11
    else:
        return cos_score
    
def load_neural_model(lang, model_path='neural-model/'):
    model = Word2Vec.load(model_path+'wmt-metrics.'+lang+'.w2v')
    bigram_transformer = Phrases(model_path+'bigram_transformer.'+lang+'.pk')
    return bigram_transformer, model

def src_trg_sent_generator(direction, dataset, quest_data_path,
                           src_bigram_transformer, trg_bigram_transformer):
    # Loads quest training files
    srcfile = quest_data_path+direction+'_source.' + dataset
    trgfile = quest_data_path+direction+'_target.' + dataset

    with io.open(srcfile, 'r') as srcfin,  \
    io.open(trgfile, 'r') as trgfin:
        # Phrasalize the sentences.
        src_sents = src_bigram_transformer[[tokenize(i.strip()) for i in srcfin]]
        trg_sents = trg_bigram_transformer[[tokenize(i.strip()) for i in trgfin]]
        for srcline, trgline in zip(src_sents, trg_sents):
            yield srcline, trgline


def create_cosine_feature(direction, dataset, quest_data_path='quest/'):
    ''' 
    # USAGE:
    direction = 'en-de'
    dataset = 'test'
    for i in create_cosine_feature(direction, dataset):
        print i
    direction = 'en-de'
    dataset = 'training'
    for i in create_cosine_feature(direction, dataset):
        print i
    '''
    # Load neural models for src and target.
    srclang, trglang = direction.lower().split('-')
    src_bigram_transformer, src_model = load_neural_model(srclang)
    trg_bigram_transformer, trg_model = load_neural_model(trglang)
    
    sent_generator = src_trg_sent_generator(direction, dataset, quest_data_path,
                                            src_bigram_transformer,
                                            trg_bigram_transformer)
    for srcline, trgline in sent_generator:
        # Tokenizes the characters list of list... 
        # TODO: merge this with the phrasalized step
        src_sent = ["".join(i) for i in list(srcline)] 
        trg_sent = ["".join(i) for i in list(trgline)]
        # Create sentence vectors
        src_vector = sent_vectorizer(src_sent, src_model)
        trg_vector = sent_vectorizer(trg_sent, trg_model)
        # Extract cosines 
        yield cosine(src_vector, trg_vector)
    

def cosine_feature(direction, dataset):
    x = np.array(list(create_cosine_feature(direction, dataset)))
    where_are_NaNs = np.isnan(x)
    x[where_are_NaNs] = 9.9999e-11
    return x.reshape(len(x), 1)
            
def create_complexity_feature(direction, dataset, quest_data_path='quest/'):
    # Load neural models for src and target.
    srclang, trglang = direction.lower().split('-')
    src_bigram_transformer, src_model = load_neural_model(srclang)
    trg_bigram_transformer, trg_model = load_neural_model(trglang)
    
    sent_generator = src_trg_sent_generator(direction, dataset, quest_data_path,
                                            src_bigram_transformer,
                                            trg_bigram_transformer)
    for srcline, trgline in sent_generator:
        # Tokenizes the characters list of list... 
        # TODO: merge this with the phrasalized step
        src_sent = ["".join(i) for i in list(srcline)] 
        trg_sent = ["".join(i) for i in list(trgline)]
        # Create sentence vectors
        src_vector = sent_vectorizer(src_sent, src_model)
        trg_vector = sent_vectorizer(trg_sent, trg_model)
        yield src_vector, trg_vector

def complexity_feature(direction, dataset):
    x1, x2 = zip(*list(create_complexity_feature(direction, dataset)))
    x = np.concatenate((x1, x2), axis=1)
    where_are_NaNs = np.isnan(x)
    x[where_are_NaNs] = 9.9999e-11
    return x