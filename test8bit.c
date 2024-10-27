// example.c
#include <stdio.h>
#include <stdint.h>  // For using int8_t type

void print_int8_array(int8_t *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("Element %d: %d\n", i, array[i]);
    }
}
