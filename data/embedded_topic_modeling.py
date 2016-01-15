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
from scipy.misc import logsumexp

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

    n_train = int(1 * len(tags))

    tweets_train, tweets_test = tokenized_tweets[:n_train], tokenized_tweets[n_train:]
    tags_train, tags_test = tags[:n_train], tags[n_train:]

    tags_to_models = {}
    for tag in tag_set: 
        tweets_for_tag = _tweets_with_tags(tweets_train, tags_train, [tag])
        # train up model for this tag
        m = copy.deepcopy(base_model)
        print("training language model for tag %s with %s examples..." % (tag, len(tweets_for_tag)))
        m.train(tweets_for_tag)
        print("ok.")
        tags_to_models[tag] = m 

    return tags_to_models
    '''
    ### now see how we do?
    preds = []
    for test_tweet in tweets_test:
        predicted_tags = tweet_preds(test_tweet, tags_to_models)
        preds.append(predicted_tags)
    '''
    

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

def _get_corpus_for_topic(tokenized_tweets, phi, topic_idx, 
                            n_samples=5000, baseline=False):
    ''' 
    here we sample topics as being present (or not) 
    in tweets with probability \propto the corresponding 
    topic probability 
    ''' 
    if topic_idx == 0 and baseline:
        return tokenized_tweets

    topic_probs = phi[:,topic_idx]
    
    normed_probs = topic_probs / topic_probs.sum()
    # sample n_samples tweets, with replacement, weighted by probs
    tweets_for_topic = np.random.choice(tokenized_tweets, p=normed_probs, 
                                            replace=True, size=n_samples)

    #tweet_topic_indicators = np.random.binomial(1, topic_probs)

    #threshold = .5
    #tweet_topic_indicators = np.array([int(tp>threshold) for tp in topic_probs])

    #tweet_idx_set = tweet_topic_indicators.nonzero()[0]
    #tweets_for_topic = [tokenized_tweets[idx] for idx in tweet_idx_set]
    #pdb.set_trace()
    return tweets_for_topic

def retrain_language_models(tokenized_tweets, phi, 
                            start_with_w2v=False, 
                            baseline=False):
    global base_model
    global w2v_model 

    if base_model is None: 
        print("initializing base model... ")

        base_model = Word2Vec(size=100, min_count=5)

        # initialize a shared vocab
        # @TODO do not need to do this everytime!
        #pdb.set_trace()
        base_model.build_vocab(tokenized_tweets)  

        '''
        currently broken! 
        '''
        if start_with_w2v and w2v_model is None: 
            print("initializing base model... ")
            path = "word-vectors/GoogleNews-vectors-negative300.bin.gz"
            w2v_model = Word2Vec.load_word2vec_format(path, binary=True)
            for word in list(base_model.vocab.keys()):
                if word in w2v_model.vocab.keys():
                    base_model[word] = w2v_model[word]
                
            
    topics_to_models = {}
    for topic_idx in range(phi.shape[1]): 
        ###
        # need to weight tweets proportionally to 
        # the prob. of their being about this topic
        topic_tweets = _get_corpus_for_topic(tokenized_tweets, phi, 
                                topic_idx, baseline=baseline)

        
        # train up model for this tag
        m = copy.deepcopy(base_model)
        m.train(topic_tweets)
        
        topics_to_models[topic_idx] = m 

    return topics_to_models

def estimate_phi(tokenized_tweets, topics_to_models, pi):
    ###
    #
    # p(topic|tweet) \propto \pi_topic * p(tweet|topic)
    #
    # for the conditional bit we use the log likelihood of each 
    # tweet under each w2v representation
    llhd = np.array(
        [np.log(pi[m_idx]) + m.score(tokenized_tweets, len(tokenized_tweets)) for 
            m_idx, m in enumerate(topics_to_models.values())]
        #[m.score(tokenized_tweets, len(tokenized_tweets)) for 
        #    m_idx, m in enumerate(topics_to_models.values())]
    )
    
    # now exponentiate to get likelihoods, subtract row max to 
    # avoid numeric overload
    lhd = np.exp(llhd - llhd.max(axis=0)) 

    # normalize
    phi = (lhd/lhd.sum(axis=0)).transpose()
    return phi 

def estimate_pi(phi):
    pi = np.zeros(phi.shape[1])
    for topic_idx in range(phi.shape[1]):
        pi[topic_idx] = phi[:,topic_idx].sum() / float(phi.shape[0])
    return pi 

def LL(topics_to_models, pi, phi, tokenized_tweets):
    ll = 0
    lg_scores = np.array([m.score(tokenized_tweets, len(tokenized_tweets)) for m 
                        in topics_to_models.values()]).transpose()
    

    for tweet_idx, tweet in enumerate(tokenized_tweets):
        #ll += logsumexp( np.log(phi[tweet_idx,:]) + lg_scores[tweet_idx,:] )
        ll += logsumexp( np.log(pi) + lg_scores[tweet_idx,:] )


    return ll 
 


def top_words_for_model(topics_to_models):
    # word_scores = [m.score([w]) for w in m.vocab]
    # V = list(m.vocab.keys())
    # topwords = np.argsort(np.array([w[0] for w in word_scores]))
    # [topics_to_models2[j].n_similarity(['vaccine'], ['autism']) for j in range(10)]
    pass 

def print_top_tweets_for_topics(phi, raw_tweets, pi, n=10, out_path=None):
    print("*** top tweets for topics ***")
    outf = None 
    if out_path is not None: 
        outf = open(out_path, 'wt')

    for topic_idx in range(phi.shape[1]):
        
        top_tweet_indices = (1-phi[:,topic_idx]).argsort()[:n]
        print("-- topic %s (pi=%s) --" % (topic_idx, pi[topic_idx]))
        if outf is not None: 
            outf.write("-- topic %s (pi=%s) --\n" % (topic_idx, pi[topic_idx]))


        for idx in top_tweet_indices:
            print(raw_tweets[idx])
            if outf is not None: 
                outf.write(raw_tweets[idx])
                outf.write("\n")
        
      

        if outf is not None: 
            outf.write("\n\n")
        print("--\n")
    print("\n***\n\n")

''' @TODO update to use "CancerReport-clean-all-data-en.txt" '''
def train_unsupervised_model(k=10, alpha=.1, max_iters=25, 
                                convergence_threshold=.001, 
                                baseline=False, silent=False):
    '''
    phi is the documents-to-topics matrix  
    '''
    global base_model
    base_model = None

    D = snowball.read_data()
    raw_tweets = D['tweet_text']

    ## 12/1 -- filter?
    tags = [snowball.which_tags(t) for t in raw_tweets]
    raw_tweets = [t for i,t in enumerate(raw_tweets) if 
                        not "#crc" in tags[i]]

    tokenized_tweets = [word_tokenize(tw) for tw in raw_tweets]
   
    def _seems_to_be_about_soccer(tweet):  
        terms = ["worldcup", "ger", "usavcrc", "fra", "italia", 
                    "mexvcrc", "#mexvcrc", "nedvscrc", "#nedvscrc", 
                    "nedcrc", "#nedcrc", "itavscrc", "#itavscrc", 
                    "uruvscrc", "#uruvscrc", "worldcup2014", "#worldcup2014",
                    "uruguay"]
        return any([t.lower() in terms for t in tweet])
        
    indices_to_keep = [idx for idx in range(len(tokenized_tweets)) if 
                            not _seems_to_be_about_soccer(tokenized_tweets[idx])]
    raw_tweets = [raw_tweets[idx] for idx in indices_to_keep]
    tokenized_tweets = [tokenized_tweets[idx] for idx in indices_to_keep]

    n = len(tokenized_tweets)
    alphas = [alpha]*k
    phi = np.zeros((n, k))

    for i in range(n):
        # initialize doc rows
        phi[i,:] = np.random.dirichlet(alphas) 

    # initial topic probability estimates
    pi = estimate_pi(phi)

    if not silent:
        print("initial assignments (random)...")
        print_top_tweets_for_topics(phi, raw_tweets, pi)

    iter_ = 0
    converged = False 

    while not converged and iter_ < max_iters:

        #######
        # 1. update language models 
        #######
        topics_to_models = retrain_language_models(tokenized_tweets, phi, baseline=baseline)

        #######
        # 2. re-estimate \phi
        #######
        phi = estimate_phi(tokenized_tweets, topics_to_models, pi)
        pi = estimate_pi(phi)

        #######
        # assess convergence
        #######
        if not silent:
            print_top_tweets_for_topics(phi, raw_tweets, pi, n=20)
        cur_LL = LL(topics_to_models, pi, phi, tokenized_tweets)
        print("finished iter: %s; LL: %s" % (iter_, cur_LL))
        #print("finished iter: %s" % iter_)
        print("\n")
        iter_ += 1

    # idx0 = (-1* phi[:,0]).argsort()[:50]
    return raw_tweets, tokenized_tweets, phi, pi, topics_to_models
