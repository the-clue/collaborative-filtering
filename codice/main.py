import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings_dataset=pd.read_csv('./../ml-latest-small/ratings.csv', usecols=["userId", "movieId", "rating", "timestamp"])

#print(ratings_dataset.head())

rating_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='rating')

user_mean_ratings = rating_matrix.mean(axis=1)
adjusted_rating_matrix = rating_matrix.subtract(user_mean_ratings, axis=0)

#print(rating_matrix)
#print(adjusted_rating_matrix)
#print(user_mean_ratings)

# Crea una Item-Similarity Matrix utilizzando una funzione di cosine_similarity predefinita
def generate_item_similarity_matrix():
    
    print("Loading Item-Similarity Matrix...")
    item_similarity_matrix_file_name = 'item_similarity_matrix'
    try:
        item_similarity_matrix = pd.read_pickle(item_similarity_matrix_file_name + '.pkl')
    except:
        print("Item-Similarity Matrix does not exist. Generating Item-Similarity Matrix...")

        adjusted_rating_matrix_T = adjusted_rating_matrix.T.fillna(0) # traspongo per ottenere una matrice item-user, necessario per cosine_similarity
        item_similarity_matrix = pd.DataFrame(cosine_similarity(adjusted_rating_matrix_T, dense_output=False), index=adjusted_rating_matrix.columns, columns=adjusted_rating_matrix.columns)
        item_similarity_matrix.to_pickle(item_similarity_matrix_file_name + '.pkl')
        #item_similarity_matrix.to_csv(item_similarity_matrix_file_name + '.csv')
    return item_similarity_matrix

def generate_item_similarity_matrix2(k):
    pass

def generate_item_similarity_matrix3():
    print("Loading Item-Similarity Matrix...")
    item_similarity_matrix_file_name = 'item_similarity_matrix.pkl'
    try:
        item_similarity_matrix = pd.read_pickle(item_similarity_matrix_file_name)
    except:
        print("Item-Similarity Matrix does not exist. Generating Item-Similarity Matrix...")
        item_similarity_matrix = pd.DataFrame(index=adjusted_rating_matrix.columns, columns=adjusted_rating_matrix.columns)
        i = 0
        for item1_id in adjusted_rating_matrix.columns:
            for item2_id in adjusted_rating_matrix.columns:
                if item1_id != item2_id:
                    item_similarity_matrix.loc[item1_id, item2_id] = adjusted_cosine_similarity(item1_id, item2_id)
                else:
                    item_similarity_matrix.loc[item1_id, item2_id] = np.nan
            i += 1
            if i % 100 == 0:
                print(f'progress: {i / len(adjusted_rating_matrix.columns)}%')

        item_similarity_matrix.to_pickle(item_similarity_matrix_file_name)
        #item_similarity_matrix.to_csv(item_similarity_matrix_file_name[:-3] + 'csv')
    return item_similarity_matrix

def adjusted_cosine_similarity(item1_id, item2_id):
    # La funzione utilizza la matrice normalizzata per evitare una sottrazione ad ogni passaggio
    item1_users = adjusted_rating_matrix[item1_id].dropna().index
    item2_users = adjusted_rating_matrix[item2_id].dropna().index

    co_rating_users = item1_users.intersection(item2_users)

    if len(co_rating_users) == 0: # forse è meglio < x, per non considerare item con troppi pochi user
        return 0  # Nessun user in comune, correlazione nulla

    numerator = 0
    denominator_1 = 0
    denominator_2 = 0

    for user_id in co_rating_users:
        item1_rating = adjusted_rating_matrix.loc[user_id, item1_id]
        item2_rating = adjusted_rating_matrix.loc[user_id, item2_id]

        numerator += item1_rating * item2_rating
        denominator_1 += item1_rating ** 2 
        denominator_2 += item2_rating ** 2 

    similarity = 0
    denominator = (denominator_1 ** 0.5) * (denominator_2 ** 0.5)
    if denominator != 0:
        similarity = numerator / denominator

    return similarity

# Funzione per predire il rating di un item (film) dato un utente considerando i k item vicini (50 per default)
def prediction_item_based(self, user_id, item_id, k=50):
    if not pd.isna(user_item_matrix.loc[user_id, item_id]): # Se l'utente ha già valutato l'item, allora non continuare
        return -1

    items_user_rated = user_item_matrix.loc[user_id].dropna().index # Considero solo gli item valutati dall'utente
    similarities = item_similarity_matrix_predefined[item_id].loc[items_user_rated] # Considero le similarità tra l'item e gli item valutati dall'utente
    nearest_neighbors = similarities.sort_values(ascending=False).head(k) # Considero i 50 neighbor più simili all'item, anche se è poco probabile che sia > 50

    weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
    similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

    for neighbor in nearest_neighbors.index:
        rating = user_item_matrix.loc[user_id, neighbor] # non ho capito se la formula considera già i rating deviati
        # rating = normalized_user_item_matrix.loc[user_id, neighbor]
        similarity = nearest_neighbors[neighbor]

        weighted_sum += similarity * rating
        similarity_sum += similarity

    if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
        return 0

    predicted_rating = weighted_sum / similarity_sum

    return predicted_rating


item_similarity_matrix = generate_item_similarity_matrix()

print(item_similarity_matrix.head())

print(item_similarity_matrix[1])
"""

import matplotlib.pyplot as plt

# Istogramma utilizzando Matplotlib
plt.hist(item_similarity_matrix[193573], bins=100, edgecolor='k')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

"""

print(ratings_dataset[ratings_dataset['movieId'] == 193573]['timestamp'])


val1 = adjusted_cosine_similarity(5, 6)

val2 = item_similarity_matrix.loc[5, 6]

print(val1, '->', val2)