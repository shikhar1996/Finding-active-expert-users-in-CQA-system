import re
import lda
import operator
import pandas as pd
from datetime import datetime
from statistics import mean


# return datetime object
def get_date(date_string):
    searchObj = re.search(r'(.*)\..*', date_string)
    date = searchObj.group(1)
    date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
    return date


# return difference in days
def date_difference(base_date, date):
    date = get_date(date)
    base_date = get_date(base_date)
    
    difference = (base_date - date).days
    return difference


joined = pd.read_pickle('movies_joined')
user_ids = joined['Id_x'].unique()

processor = lda.PreProcess()
model = lda.LDA()
model.load_model('movies_20')

n_topics = model.ntopics()
col = ['Id']+[i for i in range(n_topics)]
user_topics = pd.DataFrame(columns=col)

thresh_posts = 10
expert_weight = 0.8

now = '2018-09-27T13:08:45.277'
last_year = get_date('2017-09-27T13:08:45.277')


for _id in user_ids:
    frame = joined.loc[joined['Id_x'] == _id]
    if frame.shape[0]< thresh_posts:
        continue
    temp_array = [0]*(n_topics+1)
    temp_array[0] = _id
#-------------------------------------------------------------
# check activeness here
    # check last access date
    min_difference = 365
    date = frame.iloc[0]['LastAccessDate']
    if date is None:
        print 'No date'
        continue
    difference = date_difference(now, date)
    if difference > min_difference:
        continue    
    
    differences = []
    for j in range(frame.shape[0]-1):
        this_post = get_date(frame.iloc[j]['CreationDate_y'])
        if this_post < last_year: 
            continue
        next_post = get_date(frame.iloc[j+1]['CreationDate_y'])
        diff = (next_post - this_post).days
        differences.append(diff) 

    if len(differences) == 0:
        continue
    rate_post = mean(differences)
    active_score = float(365 - rate_post)/365

#--------------------------------------------------------------
# Score each user on each topic
    for i in range(frame.shape[0]):
        # Look only at answers posted by the user
        if frame.iloc[i]['PostTypeId'] == '1':
            continue
        if frame.iloc[i]['Body'] is not None:
            text = frame.iloc[i]['Body'].lower()
            text = processor.lemmatize(processor.remove_stopwords(text))
            dist = model.get_distribution(text) 
            topic = int(max(dist, key=lambda x:x[1])[0])
            temp_array[topic+1] += float(int(frame.iloc[i]['Score']))

    for i in range(1, len(temp_array)):
        temp_array[i] /= frame.shape[0]
        temp_array[i] *= expert_weight 
        temp_array[i] += (1-expert_weight)*active_score

    user_topics.loc[user_topics.shape[0]] = temp_array 
user_topics.to_pickle('movie_experts_20.pkl')

