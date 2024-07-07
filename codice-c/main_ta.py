import pandas as pd
import numpy as np
from ctypes import CDLL, POINTER, c_float, c_int
import matplotlib.pyplot as plt

ratings_dataset = pd.read_csv('./../ml-latest-small/ratings.csv', usecols=["userId", "movieId", "rating", "timestamp"])

rating_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='rating')
timestamp_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='timestamp')

user_mean_ratings = rating_matrix.mean(axis=1)
adjusted_rating_matrix = rating_matrix.subtract(user_mean_ratings, axis=0)

# Crea una Item-Similarity Matrix attraverso una funzione scritta in C
def generate_item_similarity_matrix():
    item_similarity_matrix_file_name = 'item_similarity_matrix_penalized'
    try:
        item_similarity_matrix = pd.read_pickle(item_similarity_matrix_file_name + '.pkl')
    except:
        print("Item-Similarity Matrix does not exist. Generating Item-Similarity Matrix...")

        # caricamento del dll
        c_code = CDLL('./script_penalized.dll')

        # assegnazione dei tipi alla funzione del dll
        c_code.function.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int]
        c_code.function.restype = c_int

        item_similarity_matrix = pd.DataFrame(index=adjusted_rating_matrix.columns, columns=adjusted_rating_matrix.columns)

        n_cols = len(adjusted_rating_matrix.columns)

        # appiattimento della matrice di similarità
        res_array = item_similarity_matrix.to_numpy(dtype=np.float32).flatten()
        res_array_arg = res_array.ctypes.data_as(POINTER(c_float))

        # appiattimento della matrice dei rating aggiustata
        array = adjusted_rating_matrix.to_numpy(dtype=np.float32).flatten()
        array_arg = array.ctypes.data_as(POINTER(c_float))

        penalty_constant = 5

        # popolazione della matrice di similarità appiattita
        c_code.function(array_arg, res_array_arg, n_cols, len(array), penalty_constant)

        item_similarity_matrix.loc[:, :] = res_array.reshape((n_cols, n_cols))

        item_similarity_matrix.to_pickle(item_similarity_matrix_file_name + '.pkl')

    return item_similarity_matrix

# funzione di test che calcola la adjusted cosine similarity tra due item (molto lenta)
def adjusted_cosine_similarity(item1_id, item2_id):
    item1_users = adjusted_rating_matrix[item1_id].dropna().index
    item2_users = adjusted_rating_matrix[item2_id].dropna().index

    co_rating_users = item1_users.intersection(item2_users)

    if len(co_rating_users) == 0:
        return 0

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
def prediction_item_based(user_id, item_id, k=50):
    if not pd.isna(rating_matrix.loc[user_id, item_id]):
        print("no prediction needed")
        return rating_matrix.loc[user_id, item_id]

    items_user_rated = rating_matrix.loc[user_id].dropna().index # Considero solo gli item valutati dall'utente
    similarities = item_similarity_matrix[item_id].loc[items_user_rated] # Considero le similarità tra l'item e gli item valutati dall'utente
    nearest_neighbors = similarities.sort_values(ascending=False).head(k) # Considero i 50 neighbor più simili all'item, anche se è poco probabile che sia > 50

    weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
    similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

    for neighbor in nearest_neighbors.index:
        rating = rating_matrix.loc[user_id, neighbor] # non ho capito se la formula considera già i rating deviati
        # rating = normalized_user_item_matrix.loc[user_id, neighbor]
        similarity = nearest_neighbors[neighbor]

        weighted_sum += similarity * rating
        similarity_sum += similarity

    if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
        return 0

    predicted_rating = weighted_sum / similarity_sum

    return predicted_rating

def prediction_item_based_with_time_decay(user_id, item_id, k=50, alpha=0.01):
    if not pd.isna(rating_matrix.loc[user_id, item_id]):
        print("no prediction needed")
        return rating_matrix.loc[user_id, item_id]

    items_user_rated = rating_matrix.loc[user_id].dropna().index # Considero solo gli item valutati dall'utente
    similarities = item_similarity_matrix[item_id].loc[items_user_rated] # Considero le similarità tra l'item e gli item valutati dall'utente
    nearest_neighbors = similarities.sort_values(ascending=False).head(k) # Considero i 50 neighbor più simili all'item, anche se è poco probabile che sia > 50

    weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
    similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

    for neighbor in nearest_neighbors.index:
        rating = rating_matrix.loc[user_id, neighbor] # non ho capito se la formula considera già i rating deviati
        # rating = normalized_user_item_matrix.loc[user_id, neighbor] # perché se è così allora va usato questo
        
        # decadimento
        rating_time = pd.to_datetime(timestamp_matrix.loc[user_id, neighbor], unit='s')
        current_time = pd.Timestamp.now()
        decay_rate = (current_time - rating_time).days
        decay_weight = np.exp(-alpha * decay_rate)

        similarity = nearest_neighbors[neighbor] * decay_weight

        weighted_sum += similarity * rating
        similarity_sum += similarity

    if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
        return 0

    predicted_rating = weighted_sum / similarity_sum

    return predicted_rating

item_similarity_matrix = generate_item_similarity_matrix()

def show_histogram():
    # Istogramma utilizzando Matplotlib
    plt.hist(item_similarity_matrix[2], bins=100, edgecolor='k')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarities')
    plt.show()

val1 = adjusted_cosine_similarity(1, 2)
val2 = adjusted_cosine_similarity(1, 3)
val3 = adjusted_cosine_similarity(1, 4)
val4 = adjusted_cosine_similarity(1, 5)

print(val1, val2, val3, val4)

val1 = item_similarity_matrix.loc[1, 2]
val2 = item_similarity_matrix.loc[1, 3]
val3 = item_similarity_matrix.loc[1, 4]
val4 = item_similarity_matrix.loc[1, 5]

print(val1, val2, val3, val4)

print("predizione utente 1 item 2:", prediction_item_based(500, 2))
print("predizione utente 1 item 2 con decadimento temporale leggero:", prediction_item_based_with_time_decay(500, 2))
print("predizione utente 1 item 2 con decadimento temporale pesante:", prediction_item_based_with_time_decay(500, 2, alpha=0.0899))

print("predizione utente 1 item 170875:", prediction_item_based(500, 170875))
print("predizione utente 1 item 170875 con decadimento temporale leggero:", prediction_item_based_with_time_decay(500, 170875))
print("predizione utente 1 item 170875 con decadimento temporale pesante:", prediction_item_based_with_time_decay(500, 170875, alpha=0.0899))

show_histogram()