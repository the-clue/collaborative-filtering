from collaborative_filtering import *

def main():
    collaborative_filterer = Collaborative_Filtering(ratings_dataset=pd.read_csv('datasets/ratings.csv', usecols=["userId", "movieId", "rating", "timestamp"]))

    timeAwareAlpha = 0.085 # sopra 0.085 il risultato è quasi sempre 0
    items_dataset = pd.read_csv('datasets/movies.csv', usecols=["movieId", "title", "genres"])

    def get_item_name_from_id(item_id):
        return items_dataset.loc[items_dataset['movieId'] == item_id, 'title'].values[0]

    print()
    while(True):
        print("Mode 1: Predict item rating for a user using Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 2: Recommend top items for a user using Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 3: Predict item rating for a user using Penalized Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 4: Recommend top items for a user using Penalized Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 5: Predict item rating for a user using Time-aware Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 6: Recommend top items for a user using Time-aware Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 7: Predict item rating for a user using Penalized Time-aware Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 8: Recommend top items for a user using Penalized Time-aware Item-based Nearest-Neighbor Collaborative Filtering")
        print("Mode 0: Exit")
        mode = int(input("Insert mode: "))

        match mode:
            case 1:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction_item_based(user_id, item_id)
                print(f'Predicted rating for user {user_id} on item "{get_item_name_from_id(item_id)}": {predicted_rating}')
            case 2:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations_item_based(user_id)
                print(f'Predicted best rated items for user {user_id}: ')
                i = 1
                for item_id, predicted_rating in recommendations.items():
                    print(f'{i}) {get_item_name_from_id(item_id)}, predicted rating: {predicted_rating}')
                    i += 1
            case 3:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction_item_based(user_id, item_id, isPenalized=True)
                print(f'Predicted rating for user {user_id} on item {get_item_name_from_id(item_id)}: {predicted_rating}')
            case 4:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations_item_based(user_id, isPenalized=True)
                print(f'Predicted best rated items for user {user_id}: ')
                i = 1
                for item_id, predicted_rating in recommendations.items():
                    print(f'{i}) {get_item_name_from_id(item_id)}, predicted rating: {predicted_rating}')
                    i += 1
            case 5:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction_item_based(user_id, item_id, timeAwareAlpha=timeAwareAlpha)
                print(f'Predicted rating for user {user_id} on item {get_item_name_from_id(item_id)}: {predicted_rating}')
            case 6:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations_item_based(user_id, timeAwareAlpha=timeAwareAlpha)
                print(f'Predicted best rated items for user {user_id}: ')
                i = 1
                for item_id, predicted_rating in recommendations.items():
                    print(f'{i}) {get_item_name_from_id(item_id)}, predicted rating: {predicted_rating}')
                    i += 1
            case 7:
                user_id = int(input("Insert User ID: "))
                item_id = int(input("Insert Item ID: "))
                predicted_rating = collaborative_filterer.prediction_item_based(user_id, item_id, isPenalized=True, timeAwareAlpha=timeAwareAlpha)
                print(f'Predicted rating for user {user_id} on item {get_item_name_from_id(item_id)}: {predicted_rating}')
            case 8:
                user_id = int(input("Insert User ID: "))
                recommendations = collaborative_filterer.predict_best_recommendations_item_based(user_id, isPenalized=True, timeAwareAlpha=timeAwareAlpha)
                print(f'Predicted best rated items for user {user_id}: ')
                i = 1
                for item_id, predicted_rating in recommendations.items():
                    print(f'{i}) {get_item_name_from_id(item_id)}, predicted rating: {predicted_rating}')
                    i += 1
            case 0:
                print("Exiting...")
                exit()
            case _:
                print("Invalid mode inserted. Try again.")
        print()

if __name__ == "__main__":
    main()