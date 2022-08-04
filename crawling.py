import tweepy
import csv
import re
from textblob import TextBlob
import emoji
import contractions

api_key = 'JgjxyWz2fnf69f0QBIf3XWHTb'
api_key_secret = 'TSVN9Gg6j12Xz5D0YlSNXf2D2mQ8iI7n59mqUBjCsrFYMZeY05'
access_token = '1360977534075068416-DPOEr5YYE6cSTVUxbwtMAXlE8rwuPR'
access_token_secret = '3O6SMyvqLTpNsLQCJcs9iwbvcyJ8iJfD9yIBA7cF27SE5'

authentication = tweepy.OAuthHandler(api_key, api_key_secret)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)

tweetsPerQry = 100
maxTweets = 5000


# proses crawling
def crawling(keywords):
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
                # bersih2 username (@blabla), hapus selain alfabet dan hapus url
                tweet_bersih = ''.join(re.sub("(@[\w]+)|(\w+:\/\/\S+)", " ", tweet.full_text).split())
                # ubah emoji menjadi text
                tweet_bersih = emoji.demojize(tweet_bersih)
                # ubah perulangan huruf yang banyak menjadi dua huruf
                tweet_bersih = re.sub(r'(.)\1+', r'\1\1', tweet_bersih)
                # hapus selain abjad
                tweet_bersih = re.sub(r'[^a-zA-Z]', ' ', tweet_bersih)
                # hapus hastag
                tweet_bersih = re.sub('#+', ' ', tweet_bersih)
                # ubah singkatan menjadi kepanjangan
                tweet_bersih = contractions.fix(tweet_bersih)
                # lowercase
                tweet_bersih = tweet_bersih.lower()

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