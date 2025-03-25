#include <stdio.h>
#include <stdlib.h>
#include "linear_regression.h"

Dataset load_data(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", path);
        exit(EXIT_FAILURE);
    }

    // Count lines (excluding header)
    int capacity = 10;
    double* X = malloc(capacity * sizeof(double));
    double* y = malloc(capacity * sizeof(double));
    int n = 0;
    char line[100];

    fgets(line, sizeof(line), file); // Skip header
    while (fgets(line, sizeof(line), file)) {
        if (n >= capacity) {
            capacity *= 2;
            X = realloc(X, capacity * sizeof(double));
            y = realloc(y, capacity * sizeof(double));
        }
        sscanf(line, "%lf,%lf", &X[n], &y[n]);
        n++;
    }

    fclose(file);
    return (Dataset){X, y, n};
}

void free_data(Dataset* data) {
    free(data->X);
    free(data->y);
}