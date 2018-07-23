import numpy as np
from lightfm import LightFM
from fetch_m20 import fetch

#fetch data and format it
data = fetch()


#create model
model = LightFM(loss='warp-kos')
#train model
model.fit(data['matrix'], epochs=30, num_threads=2)



def get_recommendation(model, coo_mtrx, user_ids):

    #number of users and movies in training data
    n_items = data['matrix'].shape[1]

    #generate recommendations for each user we input
    for user in user_ids:

        known_positives = data['movies'][data['matrix'].tocsr()[user].indices]

        scores = model.predict(user, np.arange(n_items))
        top_scores = np.argsort(-scores)[:3]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

print '\n' # Get it pretty

get_recommendation(model, data, [3, 25, 450])