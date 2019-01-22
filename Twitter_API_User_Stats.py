import tweepy
import sys
sys.path.append('/Users/rahuljain/Desktop/Python/PythonPrograms/Python_Application')
from tweepy import OAuthHandler
from twitter_credentials import *

def OAuthVerifier():
    '''
    This method is used to authenticate the user and create an object of the API class
    '''
    authentication = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    authentication.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
    api = tweepy.API(authentication)
    
    return api

def getUserStatistics(user):
    '''
    This program is used to fetch the User information
    '''
    print('\nName : ', user.name)
    print('Screen Name : ', user.screen_name)
    print('ID : ', user.id)
    print('Account Creation Date and Time : ', user.created_at)
    print('Description : ', user.description)
    print('No. of followers : ', user.followers_count)
    print('No. of friends : ', user.friends_count)
    print('No. of favourite tweets  : ', user.favourites_count)
    print('No. of posted tweets : ', user.statuses_count)
    print('Associated URL : ', user.url)

def main():
    api = OAuthVerifier()
    user = api.me()
    getUserStatistics(user)
    
    ##Different User
    name = input('\nEnter the name/handle for user identification')
    try:
        user = api.get_user(name)
        getUserStatistics(user)
    except : print(name, ' user Not found')
    
if __name__ == '__main__':
    main()
    
