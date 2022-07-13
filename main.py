import sqlite3
import tweepy
import csv
import re
import contractions
from textblob import TextBlob
from flask import Flask, request, jsonify
import socket
from nltk.stem import WordNetLemmatizer


api_key = 'JgjxyWz2fnf69f0QBIf3XWHTb'
api_key_secret = 'TSVN9Gg6j12Xz5D0YlSNXf2D2mQ8iI7n59mqUBjCsrFYMZeY05'
access_token = '1360977534075068416-DPOEr5YYE6cSTVUxbwtMAXlE8rwuPR'
access_token_secret = '3O6SMyvqLTpNsLQCJcs9iwbvcyJ8iJfD9yIBA7cF27SE5'

authentication = tweepy.OAuthHandler(api_key, api_key_secret)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)

tweetsPerQry = 100
maxTweets = 5000
keywords = ["#windows11", "windows 11"]

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)
@app.route('/' , methods=['GET','POST'])
def dashboar():
    data = {
        "datatesting" : "200",
        "datatraining" : "800"
    }
    return data

def db_connection():
    conn = None
    try:
        conn =sqlite3.connect("Database/mknn.sqlite")
    except sqlite3.error as e:
        print(e)
    return conn

# insert data training dari file csv ke tabel db
def db_import_data_training(csv):
    conn = db_connection()
    cursor = conn.cursor()

    file = open(csv)
    contents =csv.reader(file, delimiter=';')

    sql_query = """INSERT INTO training(username, tweet_asli, polaritas)VALUES(?,?,?)"""

    cursor.executemany(sql_query,contents)
    conn.commit()
    print("Success import data testing from csv")

# insert data testing dari file csv ke tabel db
def db_import_data_testing(tweet):
    conn = db_connection()
    cursor = conn.cursor()

    file = open(tweet)
    contents =tweet.reader(file, delimiter=';')

    sql_query = """INSERT INTO testing(username, tweet, polaritas_awal)VALUES(?,?,?)"""

    cursor.executemany(sql_query,contents)
    conn.commit()
    print("Success import data testing from csv")

# tampil data hasil insert
def db_get_all_training():
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT * FROM training"

    row = cursor.execute(sql_query).fetchall()

    for r in row:
        print(r)

# proses preprocessing
def preprocessing(tweet):
    # bersih2 username (@blabla), hapus selain alfabet dan hapus url
    tweet_bersih = ' '.join(
        re.sub("(@[A-Za-z0-9]+)|(#+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.full_text).split())

    # ubah perulangan huruf yang banyak menjadi dua huruf
    tweet_bersih1 = re.sub(r'(.)\1+', r'\1\1', tweet_bersih)

    # ubah menjadi kapital menjadi lowercase
    tweet_bersih2 = tweet_bersih1.lower()

    # ubah singkatan menjadi kepanjangan
    tweet_bersih3 = contractions.fix(tweet_bersih2)

    return tweet_bersih3

# proses pencarian kata baku
def lemmitization(word):
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize(word)

# proses crawling
def crawling():
    # cari data berdasarkan keyword
    for search_key in keywords:
        tweetCount = 0
        maxId = -1

        hasil_tweet = []
        hasil_isi_tweet = []

        # cari hingga maxtweet
        while tweetCount < maxTweets:
            if maxId <= 0:
                newTweets = api.search_tweets(
                    q=search_key, count=tweetsPerQry, result_type="recent", tweet_mode="extended", lang="en")

            newTweets = api.search_tweets(q=search_key, count=tweetsPerQry, result_type="recent",
                                          tweet_mode="extended", lang="en", max_id=str(maxId - 1))

            if not newTweets:
                print("Tweet Habis")
                break

            for tweet in newTweets:
                dictTweet = {}
                dictTweet["username"] = tweet.user.screen_name

                # bersih2 regex dan url
                tweet_bersih = ' '.join(
                    re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.full_text).split())

                analysis = TextBlob(tweet_bersih)

                # penghapusan retweet
                if tweet_bersih.startswith("RT"):
                    continue
                elif tweet_bersih not in hasil_isi_tweet:
                    # pemberian sentimen
                    dictTweet['tweet'] = tweet.full_text
                    if analysis.sentiment.polarity > 0.0:
                        dictTweet["sentimen"] = "positif"
                    elif analysis.sentiment.polarity == 0.0:
                        dictTweet["sentimen"] = "netral"
                    else:
                        dictTweet["sentimen"] = "negatif"

                    hasil_tweet.append(dictTweet)
                    hasil_isi_tweet.append(tweet_bersih)
                    tweetCount += 1

            maxId = newTweets[len(newTweets) - 1].id

        # pembuatan file csv
        with open(search_key + ".csv", 'a+', newline='', encoding="utf-8") as csv_file:
            fieldNames = ["username", "tweet", "sentimen"]
            writer = csv.DictWriter(
                csv_file, fieldnames=fieldNames, delimiter=",", )
            for tweet in hasil_tweet:
                writer.writerow(tweet)

if __name__ == "__main__":
    # app.run(host=my_ip)
    db_import_data_training()
    db_get_all_training()