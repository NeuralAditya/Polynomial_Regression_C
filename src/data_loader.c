#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "polynomial_regression.h"

Dataset load_data(const char* path, int degree) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", path);
        exit(EXIT_FAILURE);
    }

    int capacity = 10;
    double* raw_X = malloc(capacity * sizeof(double)); // Stores x-values
    double* y = malloc(capacity * sizeof(double)); // Stores y-values
    int n = 0;
    char line[100];

    fgets(line, sizeof(line), file); // Skip header
    while (fgets(line, sizeof(line), file)) {
        if (n >= capacity) {
            capacity *= 2;
            raw_X = realloc(raw_X, capacity * sizeof(double));
            y = realloc(y, capacity * sizeof(double));
        }
        sscanf(line, "%lf,%lf", &raw_X[n], &y[n]);
        n++;
    }
    fclose(file);

    // Allocate Vandermonde matrix (X)
    double** X = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        X[i] = malloc((degree + 1) * sizeof(double));
        for (int j = 0; j <= degree; j++) {
            X[i][j] = pow(raw_X[i], j);
        }
    }

    free(raw_X); // Free raw x-values (no longer needed)
    
    return (Dataset){X, y, n, degree};
}

void free_data(Dataset* data) {
    for (int i = 0; i < data->n_samples; i++) {
        free(data->X[i]);
    }
    free(data->X);
    free(data->y);
}
