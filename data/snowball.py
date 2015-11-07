

from collections import defaultdict
import csv 

import pdb
import gensim
from gensim import matutils, corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import nltk

# for language classification
import langid

tags = ["pap smear",
        "pap test",
        "HPV",
        "human papillomavirus",
        "HPV vaccination",
        "Gardasil",
        "cervical cancer",
        "#GoingToTheDoctor",
        "#WomensHealth",
        "colonoscopy",
        "cancer prevention",
        "cancer screening",
        "mammogram",
        "vaxx",
        "#fightcancer",
        "#stopcancerb4itstarts",
        "#screened",
        "#vaccinated",
        "#crc"]

def which_tags(tweet):
    tweet_lowercased = tweet.lower()
    tag_set = []
    for t in tags: 
        if t.lower() in tweet_lowercased:
            tag_set.append(t)
    if len(tag_set) == 0:
        tag_set.append("other")

    return tag_set

def read_data(path="CancerReport-clean.txt"):
    data = pd.read_csv(path, delimiter="\t")
    ''' 
    the data did not come with any header info (column names).
    so for now setting at least the column names for at least 
    the main columns of interest. 
    @note emailed the annenberg folks 10/21/15 with 
            request for headers.
    '''
    #data.columns.values[1] = "tweet"
    #data.columns.values[2] = "date"
    return data

def fit_lda(X, vocab, num_topics=10, passes=20, alpha=0.001):
    ''' fit LDA from a scipy CSR matrix (X). '''
    print("fitting lda...")
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                    passes=passes, alpha=alpha, 
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))

def gen_lda_model(toked_tweets, num_topics=10):
    dictionary = corpora.Dictionary(toked_tweets)
    gensim_corpus = [dictionary.doc2bow(tweet) for tweet in toked_tweets]
    lda = LdaModel(gensim_corpus, num_topics=num_topics,
                    passes=10, alpha=0.001, id2word=dictionary)

    return lda, gensim_corpus, dictionary 
    
def clean_data(original_path="CancerReport.txt", clean_path="CancerReport-clean.txt", 
                    en_only=True, THRESHOLD=.6):
    ''' 
    Read and clean the data originally provided data, 
    which was messy in that it often contained an inconsistent number 
    of columns, due to the tweets often containing tabs (which were 
    also being used as delimiters!). here we spit out a new file, 
    where we just skip those lines. 

    If the en_only flag is true here, we skip lines not classified
    by langid as English. 
    '''
    expected_num_cols = 40
    skipped_count = 0
    not_english = []
    out_str = [["id", "tweet", "date", "tweeter_name", "tweeter_info"]]
    cols = [0, 1, 2, 14, 16]
    with open(original_path, 'rU') as orig_data:
        csv_reader = csv.reader(orig_data, delimiter="\t")
        for line in csv_reader:
            if len(line) != expected_num_cols:
                skipped_count += 1
            else: 
                cols_of_interest = [line[j] for j in cols]
                lang_pred = langid.classify(cols_of_interest[1])
                # note that I'm including "de" here because for whatever 
                # reason langid kept making this mistake on english tweets. 
                # I think we should see relatively few actual German
                # tweets anyway.
                if en_only and lang_pred[0] not in ("en", "de", "sq") and lang_pred[1] > THRESHOLD:
                    not_english.append(cols_of_interest)
                else:
                    #if not contains_tag(cols_of_interest[1]):
                    #    pdb.set_trace()


                    out_str.append(cols_of_interest)
                    #pdb.set_trace()
            
    if en_only:
        clean_path = clean_path.replace(".txt", "-en.txt")

    #pdb.set_trace()
    with open(clean_path, 'w') as out_f:
        csv_writer = csv.writer(out_f, delimiter="\t")
        csv_writer.writerows(out_str)


def build_gensim_corpus(tweets, at_least=5, split_up_by_tag=False):
    # record frequencies
    STOP_WORDS = nltk.corpus.stopwords.words('english')

    # first tokenize
    toked_tweets = [list(gensim.utils.tokenize(t, lower=True)) for t in tweets]
    
    # counts
    frequency = defaultdict(int)
    for t in toked_tweets:
        for token in t:
            frequency[token] += 1

    # only used if we split by tags though.
    tags_to_tweets = defaultdict(list) 
    cleaned_toked = []
    for tweet_idx, tweet in enumerate(toked_tweets):
        cur_t = []

        for token in tweet:
            if (frequency[token] >= at_least and 
                not token in STOP_WORDS and
                len(token) > 1):
                    cur_t.append(token)


        if len(cur_t) > 0:
            if not split_up_by_tag:
                cleaned_toked.append(cur_t)
            else: 
                orig_tweet = tweets[tweet_idx]
                tag_set = which_tags(orig_tweet)
                for t in tag_set:
                    tags_to_tweets[t].append(cur_t)


    if split_up_by_tag:
        return tags_to_tweets

    return cleaned_toked







