# -*- coding: utf-8 -*-
import re
import nltk
import joblib
import tweepy
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import numpy as np
# from PIL import image
from wordcloud import WordCloud, STOPWORDS
import datetime
import html
 
class TwitterClient(object):
    
    #Generic Twitter Class for sentiment analysis.
    
    def __init__(self):
        
        #Class constructor or initialization method.
        
        # keys and tokens from the Twitter Dev Console
        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_token_secret = ''
         
        self.emoji_pattern = re.compile(pattern = "["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
            

    def _replace_emojis(self,t):
        '''
        This function replaces happy unicode emojis with "happy" and sad unicode emojis with "sad.
        '''
        emoji_happy = ["\U0001F600", "\U0001F601", "\U0001F602","\U0001F603","\U0001F604","\U0001F605", "\U0001F606", "\U0001F607", "\U0001F609", 
                        "\U0001F60A", "\U0001F642","\U0001F643","\U0001F923",r"\U0001F970","\U0001F60D", r"\U0001F929","\U0001F618","\U0001F617",
                        r"\U000263A", "\U0001F61A", "\U0001F619", r"\U0001F972", "\U0001F60B", "\U0001F61B", "\U0001F61C", r"\U0001F92A",
                        "\U0001F61D", "\U0001F911", "\U0001F917", r"\U0001F92D", r"\U0001F92B","\U0001F914","\U0001F910", r"\U0001F928", "\U0001F610", "\U0001F611",
                        "\U0001F636", "\U0001F60F","\U0001F612", "\U0001F644","\U0001F62C","\U0001F925","\U0001F60C","\U0001F614","\U0001F62A",
                        "\U0001F924","\U0001F634", "\U0001F920", r"\U0001F973", r"\U0001F978","\U0001F60E","\U0001F913", r"\U0001F9D0"]

        emoji_sad = ["\U0001F637","\U0001F912","\U0001F915","\U0001F922", r"\U0001F92E","\U0001F927", r"\U0001F975", r"\U0001F976", r"\U0001F974",
                            "\U0001F635", r"\U0001F92F", "\U0001F615","\U0001F61F","\U0001F641", r"\U0002639","\U0001F62E","\U0001F62F","\U0001F632",
                            "\U0001F633", r"\U0001F97A","\U0001F626","\U0001F627","\U0001F628","\U0001F630","\U0001F625","\U0001F622","\U0001F62D",
                            "\U0001F631","\U0001F616","\U0001F623"	,"\U0001F61E","\U0001F613","\U0001F629","\U0001F62B", r"\U0001F971",
                            "\U0001F624","\U0001F621","\U0001F620", r"\U0001F92C","\U0001F608","\U0001F47F","\U0001F480", r"\U0002620"]

        words = t.split()
        reformed = []
        for w in words:
            if w in emoji_happy:
                reformed.append("happy")
            elif w in emoji_sad:
                reformed.append("sad") 
            else:
                reformed.append(w)
        t = " ".join(reformed)
        return t

    def _replace_smileys(self,t):
        '''
        This function replaces happy smileys with "happy" and sad smileys with "sad.
        '''
        emoticons_happy = set([':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':D',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'])

        emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('])  

        words = t.split()
        reformed = []
        for w in words:
            if w in emoticons_happy:
                reformed.append("happy")
            elif w in emoticons_sad:
                reformed.append("sad") 
            else:
                reformed.append(w)
        t = " ".join(reformed)
        return t

    def _replace_contractions(self,t):
        '''
        This function replaces english lanuage contractions like "shouldn't" with "should not"
        '''
        cont = {"aren't" : 'are not', "can't" : 'cannot', "couln't": 'could not', "didn't": 'did not', "doesn't" : 'does not',
        "hadn't": 'had not', "haven't": 'have not', "he's" : 'he is', "she's" : 'she is', "he'll" : "he will", 
        "she'll" : 'she will',"he'd": "he would", "she'd":"she would", "here's" : "here is", 
        "i'm" : 'i am', "i've"	: "i have", "i'll" : "i will", "i'd" : "i would", "isn't": "is not", 
        "it's" : "it is", "it'll": "it will", "mustn't" : "must not", "shouldn't" : "should not", "that's" : "that is", 
        "there's" : "there is", "they're" : "they are", "they've" : "they have", "they'll" : "they will",
        "they'd" : "they would", "wasn't" : "was not", "we're": "we are", "we've":"we have", "we'll": "we will", 
        "we'd" : "we would", "weren't" : "were not", "what's" : "what is", "where's" : "where is", "who's": "who is",
        "who'll" :"who will", "won't":"will not", "wouldn't" : "would not", "you're": "you are", "you've":"you have",
        "you'll" : "you will", "you'd" : "you would", "mayn't" : "may not"}
        words = t.split()
        reformed = []
        for w in words:
            if w in cont:
                reformed.append(cont[w])
            else:
                reformed.append(w)
        t = " ".join(reformed)
        return t  

    def _remove_single_letter_words(self,t):
        '''
        This function removes words that are single characters
        '''
        words = t.split()
        reformed = []
        for w in words:
            if len(w) > 1:
                reformed.append(w)
        t = " ".join(reformed)
        return t  

    #Processing Tweets
    def preprocessTweets(self,t):
        
        # print(t)
        
        t = html.unescape(t)
        t = self._replace_smileys(t) # replace handwritten emojis with their feeling associated
        t = t.lower() # convert to lowercase
        t = self._replace_contractions(t) # replace short forms used in english  with their actual words
        t = self._replace_emojis(t) # replace unicode emojis with their feeling associated
        t = self.emoji_pattern.sub(r'', t) # remove emojis other than smiley emojis
        t = re.sub('\\\\u[0-9A-Fa-f]{4}','', t) # remove NON- ASCII characters
        t = re.sub("[0-9]", "", t) # remove numbers
        t = re.sub('#', '', t) # remove '#'
        t = re.sub('@[A-Za-z0–9]+', '', t) # remove '@'
        t = re.sub('RT[\s]+', '', t) # remove retweet 'RT'
        t = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', t) # remove links (URLs/ links)
        t = re.sub('[!"$%&\'()*+,-./:@;<=>?[\\]^_`{|}~]', '', t) # remove punctuations
        t = t.replace('\\\\', '')
        t = t.replace('\\', '')
        t = self._remove_single_letter_words(t) # removes single letter words

        # print(t)

        return t
    
    #Stemming of Tweets    
    def stem(self,tweet):
        stemmer = nltk.stem.PorterStemmer()
        tweet_stem = ''
        words = [word if(word[0:2]=='__') else word.lower() \
                 for word in tweet.split() \
                 if len(word) >= 3]
        words = [stemmer.stem(w) for w in words] 
        tweet_stem = ' '.join(words)
        return tweet_stem
    
    #Predict the sentiment
    def predict(self, tweet, classifier):
        
        #Utility function to classify sentiment of passed tweet
            
        tweet_processed = self.stem(self.preprocessTweets(tweet))
                 
        if ( ('__positive__') in (tweet_processed)):
             sentiment  = 1
             return sentiment
            
        elif ( ('__negative__') in (tweet_processed)):
             sentiment  = 0
             return sentiment       
        else:  
            X =  [tweet_processed]
            sentiment = classifier.predict(X)
            return sentiment[0], tweet_processed        

 
    def get_tweets(self,classifier, query, count = 1000):
            '''
            Main function to fetch tweets and parse them.
            '''
            # empty list to store parsed tweets
            classified_tweets = []
            processed_tweets = []
        
            try:
                # call twitter api to fetch tweets
                fetched_tweets = self.api.search(q = query, count = count)
     
                # parsing tweets one by one
                for tweet in fetched_tweets:
                    # empty dictionary to store required params of a tweet
                    parsed_tweet = {}
                    proc_tweets = {}
     
                    # saving text of tweet
                    parsed_tweet['text'] = tweet.text
                    
                    # saving sentiment of tweet
                    sentiment, stemmed_tweet = self.predict(tweet.text,classifier)

                    parsed_tweet['sentiment'] = sentiment
                    
                    proc_tweets['text'] = stemmed_tweet
                    proc_tweets['sentiment'] = sentiment
                    
                    # appending parsed tweet to tweets list
                    if tweet.retweet_count > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if parsed_tweet not in classified_tweets:
                            classified_tweets.append(parsed_tweet)
                            processed_tweets.append(proc_tweets)
                    else:
                        classified_tweets.append(parsed_tweet)
                        processed_tweets.append(proc_tweets)
     
                # return parsed tweets
                return classified_tweets, processed_tweets
     
            except tweepy.TweepError as e:
                # print error (if any)
                print("Error : " + str(e))  
                
                
    def get_tweets_user(self, classifier, screen_name):
        
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        classified_tweets = []
        processed_tweets = []
    
        try:
            # call twitter api to fetch tweets
            # fetched_tweets = self.api.search(q = query, count = count)
            user_id = self.api.get_user(screen_name).id_str
            fetched_tweets = self.api.user_timeline(user_id=user_id, screen_name=screen_name, count=100, include_rts = False)
            
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
                proc_tweets = {}
    
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                
                # saving sentiment of tweet
                sentiment, stemmed_tweet = self.predict(tweet.text,classifier)

                parsed_tweet['sentiment'] = sentiment
                
                proc_tweets['text'] = stemmed_tweet
                proc_tweets['sentiment'] = sentiment
                
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in classified_tweets:
                        classified_tweets.append(parsed_tweet)
                        processed_tweets.append(proc_tweets)
                else:
                    classified_tweets.append(parsed_tweet)
                    processed_tweets.append(proc_tweets)
    
            # return parsed tweets
            return classified_tweets, processed_tweets
    
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))  
        

def topic_mining(classifier, topic):
    
    classified_tweets, processed_tweets = twitClient.get_tweets(classifier, topic, count = 10000)
    plots(classified_tweets, processed_tweets, 1, topic=topic)
    
def user_mining(classifier, screen_name):
    
    classified_tweets, processed_tweets = twitClient.get_tweets_user(classifier, screen_name)
    plots(classified_tweets, processed_tweets, 2, screen_name=screen_name)

def plots(classified_tweets, processed_tweets, menu_choice, topic=None, screen_name=None):
    
    ntweets = [tweet for tweet in classified_tweets if tweet['sentiment'] == 0]
    ptweets = [tweet for tweet in classified_tweets if tweet['sentiment'] == 1]
    neg=(100*len(ntweets)/len(classified_tweets))
    pos=(100*len(ptweets)/len(classified_tweets))
    
    wordcloud_ntweets = " ".join(tweet['text'] for tweet in processed_tweets if tweet['sentiment'] == 0)
    wordcloud_ptweets = " ".join(tweet['text'] for tweet in processed_tweets if tweet['sentiment'] == 1)
    
    stopwords = set(STOPWORDS)
    
    print("Wordcloud of Positive Tweets:")
    cloud = WordCloud(background_color = "white", stopwords=stopwords).generate(wordcloud_ptweets)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    print("Wordcloud of Negative Tweets:")
    cloud = WordCloud(background_color = "white", stopwords=stopwords).generate(wordcloud_ntweets)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()         
    
    if menu_choice == 1 and topic is not None:
        print("Opinion Mining on: {}".format(topic))
    
    if menu_choice == 2 and screen_name is not None:
        print("General opinion of {} tweets".format(screen_name))
    
    # plotting graph
    ax1 = plt.axes()
    ax1.clear()
    xar = []
    yar = []
    x = 0
    y = 0
    for tweet in classified_tweets:
        x += 1
        if tweet['sentiment'] == 1 :
            y += 1
        elif tweet['sentiment'] == 0 :
            y -= 1
        xar.append(x)
        yar.append(y)
        

    ax1.plot(xar,yar)
    ax1.arrow(x, y, 0.5, 0.5, head_width=1.5, head_length=4, fc='k', ec='k')
    plt.title('Graph')
    plt.xlabel('Time')
    plt.ylabel('Overtime Opinion')
    plt.show()    

    # plotting piechart
    labels = 'Positive Tweets', 'Negative Tweets'
    sizes = [pos,neg]
    # exploding Negative
    explode = (0, 0.1) 
    fig1, ax2 = plt.subplots()
    ax2.pie(sizes, explode=explode, labels=labels, autopct='%2.3f%%', shadow=False, startangle=180)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.axis('equal') 
    
    if menu_choice == 1 and topic is not None:
        plt.title('Opinion of users')
    
    if menu_choice == 2 and screen_name is not None:
        plt.title("Opinion of {}'s tweets".format(screen_name))
    
    plt.show()
    
    # percentage of negative tweets
    print("Negative tweets percentage: ",neg)
    # percentage of positive tweets
    print("Positive tweets percentage: ",pos)
    
    # now = datetime.datetime.now()
    # print ("Date and Time analysed: ",str(now)) 

def main():
    
    print('Loading the Classifier...')
        
    classifier = joblib.load('Model/svmClassifier.pkl')
    # creating object of TwitterClient Class
    global twitClient
    twitClient = TwitterClient()
    
    print("######## Opinion Mining MENU ########")
    print("1.Get opinion on a topic")
    print("2.Get general opinion on a user's tweets")
    
    # calling function to get tweets
    while True:    
        
        menu_selection = int(input("Choice: ").strip())
        
        if menu_selection == 1:
            
            topic = input("Topic: ").strip()
            topic_mining(classifier, topic)
            
        elif menu_selection == 2:
            
            screen_name = input("User's screename: ").strip()
            user_mining(classifier, screen_name)
        
        
        q = int(input("Run Again? [Press 1 for Yes/ 0 for No]?"))
        if q == 0:
            break
    
if __name__ == "__main__":
    main()
        
