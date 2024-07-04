import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Collaborative_Filtering:

    def __init__(self, ratings_dataset):
        print("Initializing...")
        self.ratings_dataset = ratings_dataset
        self.user_item_matrix = ratings_dataset.pivot(index='userId', columns='movieId', values='rating')
        self.user_mean_ratings = self.user_item_matrix.mean(axis=1) # Valutazione media di ciascun utente

        # Per l'Item-based Nearest-Neighbor Collaborative Filtering, utilizzo direttamente una matrice normalizzata (o deviata?)
        # In realtà è utilizzabile anche per la User-based Nearest-Neighbor Collaborative Filtering (ma non l'ho usata)
        # Sottraggo la valutazione media di ciascun utente dalle loro valutazioni per evitare di farla dopo ogni passaggio

        self.normalized_user_item_matrix = self.user_item_matrix.subtract(self.user_mean_ratings, axis=0)

        self.user_similarity_matrix = self.generate_user_similarity_matrix()
        self.user_similarity_matrix_predefined = self.generate_user_similarity_matrix_predefined()

        # self.item_similarity_matrix = self.generate_item_similarity_matrix()
        self.item_similarity_matrix_predefined = self.generate_item_similarity_matrix_predefined()
        print("Initialization complete.")

    #######################################################
    # User-based Nearest-Neighbor Collaborative Filtering #
    #######################################################

    # Pearson Correlation Coefficient
    def pearson_correlation(self, user1_id, user2_id):
        user1_ratings = self.user_item_matrix.loc[user1_id].dropna().index
        user2_ratings = self.user_item_matrix.loc[user2_id].dropna().index

        co_rated_items = user1_ratings.intersection(user2_ratings)

        if len(co_rated_items) == 0: # forse è meglio < 3? Così non considero user con troppi pochi item in comune
            return -1 # Nessun item in comune, correlazione nulla

        user1_co_ratings = self.user_item_matrix.loc[user1_id, co_rated_items] # Ratings dell'utente 1 per gli item co-rated
        user2_co_ratings = self.user_item_matrix.loc[user2_id, co_rated_items] # Ratings dell'utente 2 per gli item co-rated

        user1_mean = self.user_mean_ratings.loc[user1_id]
        user2_mean = self.user_mean_ratings.loc[user2_id]

        correlation = 0
        denominator = np.sqrt(np.sum(np.square(user1_co_ratings - user1_mean))) * np.sqrt(np.sum(np.square(user2_co_ratings - user2_mean)))
        if denominator != 0:
            numerator = np.sum((user1_co_ratings - user1_mean) * (user2_co_ratings - user2_mean))
            correlation = numerator / denominator

        return correlation

    # Funzione che genera una matrice di similarity tra tutti gli utenti utilizzando la funzione pearson_correlation
    def generate_user_similarity_matrix(self):
        print("Loading User-Similarity Matrix...")
        user_similarity_matrix_file_name = 'user_similarity_matrix.pkl'
        try:
            user_similarity_matrix = pd.read_pickle(user_similarity_matrix_file_name)
        except:
            print("User-Similarity Matrix does not exist. Generating User-Similarity Matrix...")
            user_similarity_matrix = pd.DataFrame(index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
            for user1 in self.user_item_matrix.index:
                for user2 in self.user_item_matrix.index:
                    if user1 != user2:
                        user_similarity_matrix.loc[user1, user2] = self.pearson_correlation(user1, user2)
                    else:
                        user_similarity_matrix.loc[user1, user2] = 0 # oppure è meglio np.nan? Da verificare
            user_similarity_matrix.to_pickle(user_similarity_matrix_file_name)
            user_similarity_matrix.to_csv(user_similarity_matrix_file_name[:-3] + 'csv')
        return user_similarity_matrix
    
    # Funzione che genera una matrice di similarity tra tutti gli utenti utilizzando una funzione predefinita
    def generate_user_similarity_matrix_predefined(self):
        print("Loading User-Similarity Matrix Predefined...")
        user_similarity_matrix_file_name = 'user_similarity_matrix_predefined.pkl'
        try:
            user_similarity_matrix = pd.read_pickle(user_similarity_matrix_file_name)
        except:
            print("User-Similarity Matrix Predefined does not exist. Generating User-Similarity Matrix Predefined...")
            user_similarity_matrix = self.normalized_user_item_matrix.T.corr(method='pearson') # lo traspongo perché prende come input una item-user matrix
            user_similarity_matrix.to_pickle(user_similarity_matrix_file_name)
            user_similarity_matrix.to_csv(user_similarity_matrix_file_name[:-3] + 'csv')
        return user_similarity_matrix

    # Funzione per predire il rating di un item (film) dato un utente considerando i k vicini (50 per default)
    def prediction(self, user_id, item_id, k=50):
        if not pd.isna(self.user_item_matrix.loc[user_id, item_id]): # Se l'utente ha già valutato l'item, allora non continuare
            return -1

        # Se vuoi considerare i k utenti più vicini che hanno valutato l'item
        # users_who_rated = self.user_item_matrix[item_id].dropna().index # Considero solo gli utenti che hanno valutato l'item
        # similarities = self.user_similarity_matrix[user_id].loc[users_who_rated] # Calcolo le similarità tra l'utente e gli utenti che hanno valutato l'item
        
        similarities = self.user_similarity_matrix[user_id] # Considero le similarità tra l'utente e tutti gli altri utenti
        nearest_neighbors = similarities.sort_values(ascending=False).head(k) # Considero solo i 50 neighbor più simili all'utente

        weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
        similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

        for neighbor in nearest_neighbors.index:
            if pd.isna(self.user_item_matrix.loc[neighbor, item_id]): # Condizione per considerare solo item valutati dai neighbor
                continue

            rating = self.user_item_matrix.loc[neighbor, item_id] - self.user_mean_ratings.loc[neighbor]
            similarity = nearest_neighbors[neighbor]

            weighted_sum += similarity * rating
            similarity_sum += similarity

        if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
            return 0

        predicted_rating = self.user_mean_ratings.loc[user_id] + (weighted_sum / similarity_sum)

        return predicted_rating

    # Funzione che restituisce i migliori item (10 per default) da consigliare ad un utente
    def predict_best_recommendations_old(self, user_id, k=10):
        recommendations = pd.Series()

        for item_id in self.user_item_matrix.columns:
            if pd.isna(self.user_item_matrix.loc[user_id, item_id]): # Condizione per considerare solo item non ancora valutati dall'utente
                recommendations[item_id] = self.prediction(user_id, item_id)

        sorted_recommendations = recommendations.sort_values(ascending=False)
        top_k_recommendations = sorted_recommendations.head(k)

        return top_k_recommendations

    # Versione ottimizzata che non utilizza la funzione prediction
    def predict_best_recommendations(self, user_id, neighbors_size=50, recommendations_size=10):
        similarities = self.user_similarity_matrix[user_id] # Considero le similarità tra l'utente e tutti gli altri utenti
        nearest_neighbors = similarities.sort_values(ascending=False).head(neighbors_size) # Considero solo i 50 neighbor più simili all'utente

        recommendations = pd.Series()

        for item_id in self.user_item_matrix.columns:
            if pd.isna(self.user_item_matrix.loc[user_id, item_id]): # Per considerare solo item non valutati dall'utente
                weighted_sum = 0
                similarity_sum = 0

                for neighbor in nearest_neighbors.index:
                    if pd.isna(self.user_item_matrix.loc[neighbor, item_id]): # Per considerare solo item valutati dai neighbor
                        continue

                    rating = self.user_item_matrix.loc[neighbor, item_id] - self.user_mean_ratings.loc[neighbor]
                    similarity = nearest_neighbors[neighbor]

                    weighted_sum += similarity * rating
                    similarity_sum += similarity

                if similarity_sum != 0: # Se il denominatore è uguale a 0, non lo considero
                    recommendations[item_id] = self.user_mean_ratings.loc[user_id] + (weighted_sum / similarity_sum)

        sorted_recommendations = recommendations.sort_values(ascending=False)
        top_k_recommendations = sorted_recommendations.head(recommendations_size)

        return top_k_recommendations

    #######################################################
    # Item-based Nearest-Neighbor Collaborative Filtering #
    #######################################################

    # Adjusted Cosine Similarity Measure
    def adjusted_cosine_similarity(self, item1_id, item2_id):
        # La funzione utilizza la matrice normalizzata per evitare una sottrazione ad ogni passaggio
        item1_users = self.normalized_user_item_matrix[item1_id].dropna().index
        item2_users = self.normalized_user_item_matrix[item2_id].dropna().index

        co_rating_users = item1_users.intersection(item2_users)

        if len(co_rating_users) == 0: # forse è meglio < x, per non considerare item con troppi pochi user
            return 0  # Nessun user in comune, correlazione nulla

        numerator = 0
        denominator_1 = 0
        denominator_2 = 0

        for user_id in co_rating_users:
            item1_rating = self.normalized_user_item_matrix.loc[user_id, item1_id]
            item2_rating = self.normalized_user_item_matrix.loc[user_id, item2_id]

            numerator += item1_rating * item2_rating
            denominator_1 += item1_rating ** 2 
            denominator_2 += item2_rating ** 2 

        similarity = 0
        denominator = (denominator_1 ** 0.5) * (denominator_2 ** 0.5)
        if denominator != 0:
            similarity = numerator / denominator

        return similarity
    
    # Crea una Item-Similarity Matrix utilizzando la funzione adjusted_cosine_similarity
    def generate_item_similarity_matrix(self):
        print("Loading Item-Similarity Matrix...")
        item_similarity_matrix_file_name = 'item_similarity_matrix.pkl'
        try:
            item_similarity_matrix = pd.read_pickle(item_similarity_matrix_file_name)
        except:
            print("Item-Similarity Matrix does not exist. Generating Item-Similarity Matrix...")
            item_similarity_matrix = pd.DataFrame(index=self.normalized_user_item_matrix.columns, columns=self.normalized_user_item_matrix.columns)
            for item1_id in self.normalized_user_item_matrix.columns:
                for item2_id in self.normalized_user_item_matrix.columns:
                    if item1_id != item2_id:
                        item_similarity_matrix.loc[item1_id, item2_id] = self.adjusted_cosine_similarity(item1_id, item2_id)
                    else:
                        item_similarity_matrix.loc[item1_id, item2_id] = 0
            item_similarity_matrix.to_pickle(item_similarity_matrix_file_name)
            item_similarity_matrix.to_csv(item_similarity_matrix_file_name[:-3] + 'csv')
        return item_similarity_matrix

    # Crea una Item-Similarity Matrix utilizzando una funzione di cosine_similarity predefinita
    def generate_item_similarity_matrix_predefined(self):
        print("Loading Item-Similarity Matrix Predefined...")
        item_similarity_matrix_file_name = 'item_similarity_matrix_predefined.pkl'
        try:
            item_similarity_matrix = pd.read_pickle(item_similarity_matrix_file_name)
        except:
            print("Item-Similarity Matrix Predefined does not exist. Generating Item-Similarity Matrix Predefined...")
            cleaned_normalized_user_item_matrix = self.normalized_user_item_matrix.fillna(0) # pulizia necessario in quanto deve essere privo di valori nulli
            cleaned_normalized_user_item_matrix_T = cleaned_normalized_user_item_matrix.T # traspongo per ottenere una matrice item-user, necessario per cosine_similarity
            item_similarity_matrix = pd.DataFrame(cosine_similarity(cleaned_normalized_user_item_matrix_T), index=self.normalized_user_item_matrix.columns, columns=self.normalized_user_item_matrix.columns)
            item_similarity_matrix.to_pickle(item_similarity_matrix_file_name)
            item_similarity_matrix.to_csv(item_similarity_matrix_file_name[:-3] + 'csv')
        return item_similarity_matrix

    # Funzione per predire il rating di un item (film) dato un utente considerando i k item vicini (50 per default)
    def prediction_item_based(self, user_id, item_id, k=50):
        if not pd.isna(self.user_item_matrix.loc[user_id, item_id]): # Se l'utente ha già valutato l'item, allora non continuare
            return -1

        items_user_rated = self.user_item_matrix.loc[user_id].dropna().index # Considero solo gli item valutati dall'utente
        similarities = self.item_similarity_matrix_predefined[item_id].loc[items_user_rated] # Considero le similarità tra l'item e gli item valutati dall'utente
        nearest_neighbors = similarities.sort_values(ascending=False).head(k) # Considero i 50 neighbor più simili all'item, anche se è poco probabile che sia > 50

        weighted_sum = 0 # Per la sommatoria pesata delle similarità (numeratore)
        similarity_sum = 0 # Per la sommatoria delle similarità (denominatore)

        for neighbor in nearest_neighbors.index:
            rating = self.user_item_matrix.loc[user_id, neighbor] # non ho capito se la formula considera già i rating deviati
            # rating = self.normalized_user_item_matrix.loc[user_id, neighbor]
            similarity = nearest_neighbors[neighbor]

            weighted_sum += similarity * rating
            similarity_sum += similarity

        if similarity_sum == 0: # Nel caso si annullasse la sommatoria delle similarità
            return 0

        predicted_rating = weighted_sum / similarity_sum

        return predicted_rating

    def predict_best_recommendations_item_based(self, user_id, recommendations_size=10):
        recommendations = pd.Series()

        for item_id in self.user_item_matrix.columns:
            if pd.isna(self.user_item_matrix.loc[user_id, item_id]): # Per considerare solo item non ancora valutati dall'utente
                recommendations[item_id] = self.prediction_item_based(user_id, item_id)

        sorted_recommendations = recommendations.sort_values(ascending=False)
        top_k_recommendations = sorted_recommendations.head(recommendations_size)

        return top_k_recommendations

def main():
    collaborative_filterer = Collaborative_Filtering(ratings_dataset=pd.read_csv('ml-latest-small/ratings.csv'))

    print()
    while(True):
        # Interface
        print("Mode 1: Predict item rating for a user using User-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 2: Recommend top items for a user using User-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 3: Predict item rating for a user using Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 4: Recommend top items for a user using Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 5: Predict item rating for a user using Time-aware Collaborative Filtering")
        print("Mode 6: Recommend top items for a user using Time-aware Collaborative Filtering")
        print("Mode 0: Exit")
        mode = int(input("Insert mode: "))

        match mode:
            case 1:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction(user_id, item_id)
                if predicted_rating == 0:  # Vuol dire che gli utenti più simili all'utente target non hanno valutato l'item
                    print(f'Can not predict a rating for item {item_id} for lack of information.')
                elif predicted_rating == -1: # Vuol dire che l'utente target ha già una valutazione per l'item
                    print(f'User {user_id} has already got a rating for item {item_id}.')
                else:
                    print(f'Predicted rating for user {user_id} on item {item_id}: {predicted_rating}')
            case 2:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations(user_id)
                print(f'Predicted best rated items for user {user_id}: ')
                print(recommendations)
            case 3:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction_item_based(user_id, item_id)
                if predicted_rating == 0:  # Vuol dire che gli utenti più simili all'utente target non hanno valutato l'item
                    print(f'Can not predict a rating for item {item_id} for lack of information.')
                elif predicted_rating == -1: # Vuol dire che l'utente target ha già una valutazione per l'item
                    print(f'User {user_id} has already got a rating for item {item_id}.')
                else:
                    print(f'Predicted rating for user {user_id} on item {item_id}: {predicted_rating}')
            case 4:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations_item_based(user_id)
                print(f'Predicted best rated items for user {user_id}: ')
                print(recommendations)
            case 5:
                print("Not yet implemented.")
            case 6:
                print("Not yet implemented.")
            case 0:
                print("Exiting...")
                exit()
            case _:
                print("Invalid mode inserted. Try again.")
        print()

if __name__ == "__main__":
    main()