import pandas as pd
from sklearn.model_selection import train_test_split
from collaborative_filtering import Collaborative_Filtering

# considero come rilevanti i film per cui l'utente darebbe un rating >= di 3.5
def evaluate(collaborative_filterer, ratings_test_set, k=50, isPenalized=False, timeAwareAlpha=0.0, threshold=3.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in ratings_test_set.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']

        predicted_rating = collaborative_filterer.prediction_item_based(user_id, movie_id, k, isPenalized, timeAwareAlpha)
        actual_relevant = actual_rating >= threshold
        predicted_relevant = predicted_rating >= threshold

        if actual_relevant and predicted_relevant:
            true_positives += 1
        elif not actual_relevant and predicted_relevant:
            false_positives += 1
        elif actual_relevant and not predicted_relevant:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def main():
    ratings_dataset = pd.read_csv('datasets/ratings.csv', usecols=["userId", "movieId", "rating", "timestamp"])
    items_dataset = pd.read_csv('datasets/movies.csv', usecols=["movieId", "title", "genres"])

    ratings_training_set, ratings_test_set = train_test_split(ratings_dataset, test_size=0.2, random_state=42) # split 80-20

    collaborative_filterer = Collaborative_Filtering(ratings_dataset=ratings_training_set, items_dataset=items_dataset)

    timeAwareAlpha_strong = 0.085
    timeAwareAlpha_medium = 0.084
    timeAwareAlpha_weak = 0.08
    timeAwareAlpha_very_weak = 0.05
    timeAwareAlpha_minimum = 0.01

    precision_base, recall_base = evaluate(collaborative_filterer, ratings_test_set)
    print("=======BASE")
    print(f"Precision: {precision_base:.4f}")
    print(f"Recall: {recall_base:.4f}")
    print()

    precision_base_penalized, recall_base_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True)
    print("=======BASE - Penalized:")
    print(f"Precision: {precision_base_penalized:.4f}")
    print(f"Recall: {recall_base_penalized:.4f}")
    print()

    precision_time_aware, recall_time_aware = evaluate(collaborative_filterer, ratings_test_set, timeAwareAlpha=timeAwareAlpha_strong)
    print("=======TIME-AWARE-STRONG:")
    print(f"Precision: {precision_time_aware:.4f}")
    print(f"Recall: {recall_time_aware:.4f}")
    print()

    precision_time_aware_penalized, recall_time_aware_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True, timeAwareAlpha=timeAwareAlpha_strong)
    print("=======TIME-AWARE-STRONG - Penalized:")
    print(f"Precision: {precision_time_aware_penalized:.4f}")
    print(f"Recall: {recall_time_aware_penalized:.4f}")
    print()

    precision_time_aware, recall_time_aware = evaluate(collaborative_filterer, ratings_test_set, timeAwareAlpha=timeAwareAlpha_medium)
    print("=======TIME-AWARE-MEDIUM:")
    print(f"Precision: {precision_time_aware:.4f}")
    print(f"Recall: {recall_time_aware:.4f}")
    print()

    precision_time_aware_penalized, recall_time_aware_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True, timeAwareAlpha=timeAwareAlpha_medium)
    print("=======TIME-AWARE-MEDIUM - Penalized:")
    print(f"Precision: {precision_time_aware_penalized:.4f}")
    print(f"Recall: {recall_time_aware_penalized:.4f}")
    print()

    precision_time_aware, recall_time_aware = evaluate(collaborative_filterer, ratings_test_set, timeAwareAlpha=timeAwareAlpha_weak)
    print("=======TIME-AWARE-WEAK:")
    print(f"Precision: {precision_time_aware:.4f}")
    print(f"Recall: {recall_time_aware:.4f}")
    print()

    precision_time_aware_penalized, recall_time_aware_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True, timeAwareAlpha=timeAwareAlpha_weak)
    print("=======TIME-AWARE-WEAK - Penalized:")
    print(f"Precision: {precision_time_aware_penalized:.4f}")
    print(f"Recall: {recall_time_aware_penalized:.4f}")
    print()

    precision_time_aware, recall_time_aware = evaluate(collaborative_filterer, ratings_test_set, timeAwareAlpha=timeAwareAlpha_very_weak)
    print("=======TIME-AWARE-VERY-WEAK:")
    print(f"Precision: {precision_time_aware:.4f}")
    print(f"Recall: {recall_time_aware:.4f}")
    print()

    precision_time_aware_penalized, recall_time_aware_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True, timeAwareAlpha=timeAwareAlpha_very_weak)
    print("=======TIME-AWARE-VERY-WEAK - Penalized:")
    print(f"Precision: {precision_time_aware_penalized:.4f}")
    print(f"Recall: {recall_time_aware_penalized:.4f}")
    print()

    precision_time_aware, recall_time_aware = evaluate(collaborative_filterer, ratings_test_set, timeAwareAlpha=timeAwareAlpha_minimum)
    print("=======TIME-AWARE-MINIMUM:")
    print(f"Precision: {precision_time_aware:.4f}")
    print(f"Recall: {recall_time_aware:.4f}")
    print()

    precision_time_aware_penalized, recall_time_aware_penalized = evaluate(collaborative_filterer, ratings_test_set, isPenalized=True, timeAwareAlpha=timeAwareAlpha_minimum)
    print("=======TIME-AWARE-MINIMUM - Penalized:")
    print(f"Precision: {precision_time_aware_penalized:.4f}")
    print(f"Recall: {recall_time_aware_penalized:.4f}")
    print()

if __name__ == "__main__":
    main()