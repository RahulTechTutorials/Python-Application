import tweepy
import sys
sys.path.append('/Users/rahuljain/Desktop/Python/PythonPrograms/Python_Application')
from Twitter_API_User_Stats import OAuthVerifier
from twitter_credentials import *

def getUserFollowers(api):
    print('\nFollower Names')
    for followers in tweepy.Cursor(api.followers).items():
        print(followers.screen_name)


def getUserFriends(api):
    print('\nFriends Names')
    for friends in tweepy.Cursor(api.friends).items():
        print(friends.screen_name)

    
def getUsertweets(api):
    print('\nTweets')
    for tweet in tweepy.Cursor(api.user_timeline).items():
        print(tweet.text)


def main():
    api = OAuthVerifier()
    getUserFollowers(api)
    getUserFriends(api)
    getUsertweets(api)

if __name__ == '__main__':
    main()
    
