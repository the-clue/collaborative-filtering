#include <stdio.h>
#include <stdint.h>

int function(int* array, int length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
        printf("%d ", array[i]);
    }
    printf("\n");
    return sum;
}