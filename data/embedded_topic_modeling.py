import multiprocessing
import itertools 
import copy 
import pdb 

import gensim
import snowball
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd 
import numpy as np 

def _get_tags(tweets):
    tweet_tags = [snowball.which_tags(tw) for tw in tweets]
    unique_tag_set = list(set(itertools.chain(*tweet_tags)))
    return tweet_tags, unique_tag_set

def _tweets_with_tags(tweets, tags, target_tags):
    tweets_with_tags = []
    for tweet, tweet_tags in zip(tweets, tags):
        if any([tt in target_tags for tt in tweet_tags]):
            tweets_with_tags.append(tweet)
    return tweets_with_tags


##
# supervised variant (classification through distributed inversion)
# partially based on Taddy's code
#   https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb
def train_supervised_model():
    D = snowball.read_data()
    raw_tweets = D['tweet']
    tokenized_tweets = [word_tokenize(tw) for tw in raw_tweets]
    # as a first pass, we'll tree these as labels.
    tags, tag_set = _get_tags(raw_tweets)
    
    base_model = Word2Vec()
    #workers=multiprocessing.cpu_count(),
    #    iter=3)
    
    # initialize a shared vocab
    base_model.build_vocab(tokenized_tweets)  

    n_train = int(.75 * len(tags))
    tweets_train, tweets_test = tokenized_tweets[:n_train], tokenized_tweets[n_train:]
    tags_train, tags_test = tags[:n_train], tags[n_train:]

    tags_to_models = {}
    for tag in tag_set: 
        tweets_for_tag = _tweets_with_tags(tweets_train, tags_train, [tag])
        # train up model for this tag
        m = copy.deepcopy(base_model)
        print("training language model for tag %s..." % tag)
        m.train(tweets_for_tag)
        print("ok.")
        tags_to_models[tag] = m 


    ### now see how we do?
    preds = []
    for test_tweet in tweets_test:
        predicted_tags = tweet_preds(test_tweet, tags_to_models)
        preds.append(predicted_tags)


'''
def tweet_preds(tweet, topics_to_models):
    probs = tweet_probs(tweet, topics_to_models)
    idx = probs.loc[0].nonzero()
    return probs.columns[idx].tolist()


def tweet_probs(tweet, topics_to_models):
    # the log likelihood of each tweet under each 
    # w2v representation
    llhd = np.array( [ m.score([tweet], 1) for m in topics_to_models.values() ] )

    # now exponentiate to get likelihoods, subtract row max to 
    # avoid numeric overload
    lhd = np.exp(llhd - llhd.max(axis=0)) 

    # normalize across tag models
    probs = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose(), 
                            columns=tags_to_models.keys()  )
    
    return probs
'''

def tweet_preds(tweets, topics_to_models):
    probs = tweet_probs(tweets, topics_to_models)
    # @TODO this is a terrible way of doing things!
    preds = []
    for i in range(len(tweets)):
        print(i)
        idx = probs.loc[i].nonzero()
        preds.append(probs.columns[idx].tolist())

    return preds

def tweets_probs(tweets, topics_to_models):
    # the log likelihood of each tweet under each 
    # w2v representation
    llhd = np.array( [ m.score(tweets, len(tweets)) for m in topics_to_models.values() ] )

    # now exponentiate to get likelihoods, subtract row max to 
    # avoid numeric overload
    lhd = np.exp(llhd - llhd.max(axis=0)) 

    # normalize across tag models
    probs = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose(), 
                            columns=tags_to_models.keys()  )
    
    return probs

