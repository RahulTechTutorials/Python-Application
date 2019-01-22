import tweepy
import sys
sys.path.append('/Users/rahuljain/Desktop/Python/PythonPrograms/Python_Application')
from Twitter_API_User_Stats import OAuthVerifier
from twitter_credentials import *

class myStreamListener(tweepy.StreamListener):
    
    def on_status(self,status):
        print(status.text)
    def on_error(self,status):
        if status == 420:
            return False

def main():
    api = OAuthVerifier()
    searchKey = input('\nEnter the search key for tweets')
    listenerObj = myStreamListener()
    myStream = tweepy.Stream(api.auth,listenerObj )
    myStream.filter(track = searchKey)

if __name__ == '__main__':
    main()
    
