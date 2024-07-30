import pandas as pd
import numpy as np
from ctypes import CDLL, POINTER, c_float, c_int
import matplotlib.pyplot as plt

class Collaborative_Filtering:

    def __init__(self, ratings_dataset, items_dataset):
        self.ratings_dataset = ratings_dataset
        self.rating_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='rating')
        self.timestamp_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='timestamp')
        self.user_mean_ratings = self.rating_matrix.mean(axis=1)
        self.adjusted_rating_matrix = self.rating_matrix.subtract(self.user_mean_ratings, axis=0)

        self.item_similarity_matrix = self.generate_item_similarity_matrix()
        self.item_similarity_matrix_penalized = self.generate_item_similarity_matrix(5)

        self.items_dataset = items_dataset 
    
    # Funzione di test che calcola l'Adjusted Cosine Similarity tra due item (molto lenta)
    def adjusted_cosine_similarity(self, item1_id, item2_id):
        item1_ratings = self.adjusted_rating_matrix[item1_id].dropna()
        item2_ratings = self.adjusted_rating_matrix[item2_id].dropna()

        co_rating_users = item1_ratings.index.intersection(item2_ratings.index)

        if len(co_rating_users) == 0:
            return 0
        
        item1_ratings = item1_ratings.loc[co_rating_users]
        item2_ratings = item2_ratings.loc[co_rating_users]

        numerator = np.dot(item1_ratings, item2_ratings)
        denominator = np.sqrt(np.dot(item1_ratings, item1_ratings)) * np.sqrt(np.dot(item2_ratings, item2_ratings))

        if denominator == 0:
            return 0

        return numerator / denominator

    # Crea una Item-Similarity Matrix attraverso una funzione scritta in C, possibilmente con una penalità
    def generate_item_similarity_matrix(self, penalty_constant = 0):
        item_similarity_matrix_file_name = 'item_similarity_matrix'
        if penalty_constant > 0: # è una costante utilizzare per introddure una penalità al valore di similarità
            item_similarity_matrix_file_name += '_penalized'
        try:
            item_similarity_matrix = pd.read_pickle('data/' + item_similarity_matrix_file_name + '.pkl')
        except:
            # caricamento del dll e assegnazione dei tipi alla funzione del dll
            if penalty_constant > 0:
                c_code = CDLL('c-scripts/script_penalized.dll')
                c_code.function.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
            else:
                c_code = CDLL('c-scripts/script.dll')
                c_code.function.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int]
            c_code.function.restype = c_int

            item_similarity_matrix = pd.DataFrame(index=self.adjusted_rating_matrix.columns, columns=self.adjusted_rating_matrix.columns)

            n_cols = len(self.adjusted_rating_matrix.columns)

            # appiattimento della matrice di similarità
            res_array = item_similarity_matrix.to_numpy(dtype=np.float32).flatten()
            res_array_arg = res_array.ctypes.data_as(POINTER(c_float))

            # appiattimento della matrice dei rating aggiustata
            array = self.adjusted_rating_matrix.to_numpy(dtype=np.float32).flatten()
            array_arg = array.ctypes.data_as(POINTER(c_float))

            # popolazione della matrice di similarità appiattita
            if penalty_constant > 0:
                c_code.function(array_arg, res_array_arg, n_cols, len(array), penalty_constant)
            else:
                c_code.function(array_arg, res_array_arg, n_cols, len(array))

            item_similarity_matrix.loc[:, :] = res_array.reshape((n_cols, n_cols))

            item_similarity_matrix.to_pickle('data/' + item_similarity_matrix_file_name + '.pkl')

        return item_similarity_matrix

    # Funzione per predire il rating di un item (film) dato un utente considerando i k item vicini (50 per default)
    def prediction_item_based(self, user_id, item_id, k=50, isPenalized=False, timeAwareAlpha=0.0):
        if item_id not in self.rating_matrix.columns:  # se l'id non esiste, allora ritorna 0
            return 0

        if not pd.isna(self.rating_matrix.loc[user_id, item_id]): # Allora la predizione non è necessaria
            return self.rating_matrix.loc[user_id, item_id]

        items_user_rated = self.rating_matrix.loc[user_id].dropna().index # Considero solo gli item valutati dall'utente
        # Considero le similarità tra l'item e gli item valutati dall'utente
        if isPenalized:
            similarities = self.item_similarity_matrix_penalized[item_id].loc[items_user_rated]
        else:
            similarities = self.item_similarity_matrix[item_id].loc[items_user_rated]
        nearest_neighbors = similarities.sort_values(ascending=False)
        
        nearest_neighbors = nearest_neighbors[nearest_neighbors > 0]

        k = min(k, len(nearest_neighbors))
        if k == 0: # Caso in cui non ci sono vicini
            return 0
        nearest_neighbors = nearest_neighbors.head(k)

        weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
        similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

        for neighbor in nearest_neighbors.index:
            rating = self.rating_matrix.loc[user_id, neighbor]
            # rating = self.adjusted_rating_matrix.loc[user_id, neighbor] # Se la formula nelle slide considera la matrice adjusted
            
            decay_weight = 1
            # Decadimento
            if (timeAwareAlpha > 0):
                rating_time = pd.to_datetime(self.timestamp_matrix.loc[user_id, neighbor], unit='s')
                current_time = pd.Timestamp.now()
                decay_rate = (current_time - rating_time).days # la penalizzazione si basa sui giorni passati
                decay_weight = np.exp(-timeAwareAlpha * decay_rate)

            similarity = nearest_neighbors[neighbor] * decay_weight

            weighted_sum += similarity * rating
            similarity_sum += similarity

        if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
            return 0

        predicted_rating = weighted_sum / similarity_sum

        return predicted_rating

    def predict_best_recommendations_item_based(self, user_id, recommendations_size=10, isPenalized=False, timeAwareAlpha=0.0):
        recommendations = pd.Series()

        for item_id in self.adjusted_rating_matrix.columns:
            if pd.isna(self.adjusted_rating_matrix.loc[user_id, item_id]): # Per considerare solo item non ancora valutati dall'utente
                recommendations[item_id] = self.prediction_item_based(user_id, item_id, isPenalized=isPenalized, timeAwareAlpha=timeAwareAlpha)

        sorted_recommendations = recommendations.sort_values(ascending=False)
        top_k_recommendations = sorted_recommendations.head(recommendations_size)

        return top_k_recommendations

    # UTILITY FUNCTIONS
    def get_item_name_from_id(self, item_id):
        return self.items_dataset.loc[self.items_dataset['movieId'] == item_id, 'title'].values[0]

    def get_recent_favorite_items_of_user(self, user_id, k=10):
        user_ratings = self.ratings_dataset[self.ratings_dataset['userId'] == user_id]
        max_rating = user_ratings['rating'].max()
        favorite_items = user_ratings[user_ratings['rating'] == max_rating]
        favorite_items = favorite_items.sort_values(by='timestamp', ascending=False)
        return favorite_items.head(k)

    def get_named_recent_favorite_items_of_user(self, user_id):
        favorite_item_ids = self.get_recent_favorite_items_of_user(user_id)['movieId']
        favorite_item_names = []
        for item_id in favorite_item_ids:
            favorite_item_names.append(self.get_item_name_from_id(item_id))
        return favorite_item_names

    def show_histogram(self, isPenalized=False): # mostra una istogramma della matrice di similarità tra item
        if isPenalized:
            plt.hist(self.item_similarity_matrix_penalized[2], bins=100, edgecolor='k')
        else:
            plt.hist(self.item_similarity_matrix[2], bins=100, edgecolor='k')
        plt.xlabel('Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarities')
        plt.show()