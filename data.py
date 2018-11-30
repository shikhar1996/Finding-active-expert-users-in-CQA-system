import pandas as pd
import xml.etree.ElementTree as ET
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# convert xml to pandas dataframe
class XML2DF:
    def __init__(self, xml_source):
        self.root = ET.parse(xml_source).getroot()
        self.keys = None 

    def parse_child(self, child):
        temp = []
        for key in self.keys:
            if key in child.keys():
                temp.append(child[key])
            else:
                temp.append(None)
        return temp

    def parse(self):
        structured = []
        max_len = 0
        for index in range(len(self.root)):
            child = self.root[index].attrib
            structured.append(self.parse_child(child)) 
        return pd.DataFrame(structured, columns=self.keys)
    
    def assign_headers(self, keys):
        self.keys = keys
# -------------------------------------------------------------------------------------

xml2df = XML2DF('./Dataset/movies.stackexchange.com/Users.xml')
keys = ['Id', 'Reputation', 'CreationDate', 'DisplayName', 'EmailHash', 'LastAccessDate',
        'WebsiteUrl', 'Location', 'Age', 'AboutMe', 'Views', 'UpVotes', 'DownVotes']
xml2df.assign_headers(keys)
anime_users = xml2df.parse()

xml2df = XML2DF('./Dataset/movies.stackexchange.com/Posts.xml')
keys = ['Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 'Score',
        'ViewCount', 'Body', 'OwnerUserId', 'LastEditorUserId', 'LastEditorDisplayName',
        'LastEditDate', 'LastActivityDate', 'CommunityOwnedDate', 'ClosedDate', 
        'Title', 'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount']
xml2df.assign_headers(keys)
posts = xml2df.parse()

anime_users.to_pickle('movies_users')
posts.to_pickle('movies_posts')

# -------------------------------------------------------------------------------------

# 33 columns
joined = pd.merge(anime_users, posts, left_on='Id', right_on='OwnerUserId')
joined.to_pickle('movies_joined')


