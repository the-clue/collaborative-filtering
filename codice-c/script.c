#include <stdio.h>
#include <stdint.h>
#include <math.h>

float adjusted_cosine_similarity(float* rating_matrix, int n_cols, int length, int item1, int item2) {
    int n_rows = length / n_cols;
    float numerator = 0.0;
    float denominator1 = 0.0;
    float denominator2 = 0.0;
    float similarity = 0.0;
    for (int u = 0; u < n_rows; u++) {
        float r_u_item1 = rating_matrix[u * n_cols + item1];
        float r_u_item2 = rating_matrix[u * n_cols + item2];

        if ((r_u_item1 == r_u_item1) && (r_u_item2 == r_u_item2)) {
            numerator += r_u_item1 * r_u_item2;
            denominator1 += r_u_item1 * r_u_item1;
            denominator2 += r_u_item2 * r_u_item2;
        }
    }
    float denominator = sqrtf(denominator1) * sqrtf(denominator2);
    if (denominator != 0.0) {
        similarity = numerator / denominator;
    }
    return similarity;
}

int function(float* rating_matrix, float* item_similarity_matrix, int n_cols, int length) {
    for (int item1 = 0; item1 < n_cols; item1++) {
        for (int item2 = item1 + 1; item2 < n_cols; item2++) {
            if (item1 != item2) {
                float similarity = adjusted_cosine_similarity(rating_matrix, n_cols, length, item1, item2);
                item_similarity_matrix[item1 * n_cols + item2] = similarity;
                item_similarity_matrix[item2 * n_cols + item1] = similarity;
            }
        }
        if (item1 % 100 == 0) {
            printf("%d%%\n", item1 * 100 / n_cols);
        }
    }
    return 0;
}