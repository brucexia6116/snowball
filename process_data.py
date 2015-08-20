import pandas as pd
from sqlalchemy import create_engine
from pysqlite2 import dbapi2 as sqlite3

def read_in_data(db_path="sqlite:///snowball.sqlite"):
    engine = create_engine(db_path)
    con = engine.connect()

    ''' tweets; stored in the `data' table '''
    all_tweets = con.execute("select * from data")
    tweets_df = pd.DataFrame(all_tweets.fetchall())
    tweets_df.columns = all_tweets.keys() 

    ''' users '''
    users = con.execute("select * from users")
    users_df = pd.DataFrame(users.fetchall())
    users_df.columns = users.keys()

    ''' hashtags '''
    hashtags = con.execute("select * from hashtags")
    hashtags_df = pd.DataFrame(hashtags.fetchall())
    hashtags_df.columns = hashtags.keys()

    return tweets_df, users_df, hashtags_df



