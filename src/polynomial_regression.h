#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

// Structure for dataset
typedef struct {
    double** X;        // Vandermonde matrix (2D array)
    double* y;         // Target values
    int n_samples;     // Number of data points
    int degree;        // Degree of the polynomial
} Dataset;

// Structure for polynomial regression model
typedef struct {
    double* coefficients;  // Array to store polynomial coefficients
    int degree;            // Degree of the polynomial
} Model;

// Data handling (ONLY DECLARATION, NO IMPLEMENTATION)
Dataset load_data(const char* path, int degree);
void free_data(Dataset* data);

// Core algorithm
void train(Model* model, Dataset* data);
double predict(Model* model, double x);
void free_model(Model* model);

#endif // POLYNOMIAL_REGRESSION_H
