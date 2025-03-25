#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

typedef struct {
    double* X;
    double* y;
    int n_samples;
} Dataset;

typedef struct {
    double slope;
    double intercept;
} Model;

// Data handling
Dataset load_data(const char* path);
void free_data(Dataset* data);

// Core algorithm
void train(Model* model, Dataset* data, float lr, int epochs);
double predict(Model* model, double x);

#endif