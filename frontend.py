import lda
from user_profile import get_date
import pandas as pd


def reciprocal_rank(id_array, accepted_id):
    count = 1
    for i in range(len(id_array)):
        if id_array[i] == accepted_id:
            return 1.0/count
        count += 1
    return -1 


def mrr(rrs):
    total = 0
    for i in range(len(rrs)):
        total += rrs[i]
    return float(total)/len(rrs)


model = lda.LDA()
model.load_model('movies_20')
processor = lda.PreProcess()

joined = pd.read_pickle('movies_joined')
experts = pd.read_pickle('movie_experts.pkl')
questions = joined.loc[joined['PostTypeId'] == '1']

top_n = 20

incorrect = 0
total = 0
rrs = []
count = 0
for i in range(questions.shape[0]):
    question = questions.iloc[i]
    date = get_date(question['CreationDate_y'])
    if date < get_date('2017-09-01T13:08:34.566'):
        continue
    count += 1
    if count >= 100:
        break
    ques = questions.iloc[i]['Body']
    if not ques:
        continue

    ques = processor.lemmatize(processor.remove_stopwords(ques))        
    topic_dist = model.get_distribution(ques)
    best_topic = int(max(topic_dist, key = lambda x:x[1])[0])
    sorted_experts = experts[best_topic].sort_values(ascending=False)
    expert_ids = []

    for j in range(top_n):
        expert_id = experts.loc[experts[best_topic] == sorted_experts.iloc[j]]['Id'].iloc[0]
        expert_ids.append(expert_id)

    answer_id = questions['AcceptedAnswerId'].iloc[i]
    if answer_id is None:
        continue 

    accepted_id = joined.loc[joined['Id_y'] == answer_id]['Id_x']
    if accepted_id.shape[0] == 0:
        continue

    total += 1
    rr = reciprocal_rank(expert_ids, accepted_id.iloc[0]) 
    
    if rr == -1:
        incorrect += 1
        continue

    rrs.append(rr)
    break

print "Inaccuracy: ", float(incorrect)/total
print "MRR: ", mrr(rrs)

