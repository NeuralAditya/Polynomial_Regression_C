#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Dataset structure
typedef struct {
    double *X;      // Feature array
    double *y;      // Target array
    int n_samples;  // Number of data points
    double slope;   // Model parameter (weight)
    double intercept; // Model parameter (bias)
} LinearModel;

// Function to load data from CSV (format: x,y)
void load_data(const char *filename, LinearModel *model) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // Count lines in file
    int n = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') n++;
    }
    rewind(file);

    // Allocate memory
    model->X = (double *)malloc(n * sizeof(double));
    model->y = (double *)malloc(n * sizeof(double));
    model->n_samples = n;

    // Read data (skip header if exists)
    char header[256];
    fgets(header, sizeof(header), file); // Assumes CSV has header

    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf,%lf\n", &model->X[i], &model->y[i]);
    }
    fclose(file);
}

// Mean Squared Error (MSE) loss
double compute_loss(LinearModel *model) {
    double loss = 0;
    for (int i = 0; i < model->n_samples; i++) {
        double pred = model->slope * model->X[i] + model->intercept;
        loss += pow(pred - model->y[i], 2);
    }
    return loss / model->n_samples;
}

// Gradient Descent training
void train(LinearModel *model, double learning_rate, int epochs) {
    for (int e = 0; e < epochs; e++) {
        double grad_slope = 0, grad_intercept = 0;

        // Compute gradients
        for (int i = 0; i < model->n_samples; i++) {
            double pred = model->slope * model->X[i] + model->intercept;
            grad_slope += (pred - model->y[i]) * model->X[i];
            grad_intercept += (pred - model->y[i]);
        }

        // Update parameters
        model->slope -= learning_rate * (grad_slope / model->n_samples);
        model->intercept -= learning_rate * (grad_intercept / model->n_samples);

        // Print loss every 100 epochs
        if (e % 100 == 0) {
            printf("Epoch %d, Loss: %f\n", e, compute_loss(model));
        }
    }
}

// Save predictions to CSV for plotting
void save_predictions(LinearModel *model, const char *filename) {
    FILE *file = fopen(filename, "w");
    fprintf(file, "x,y_true,y_pred\n");
    for (int i = 0; i < model->n_samples; i++) {
        double pred = model->slope * model->X[i] + model->intercept;
        fprintf(file, "%f,%f,%f\n", model->X[i], model->y[i], pred);
    }
    fclose(file);
}

int main() {
    LinearModel model = {NULL, NULL, 0, 0, 0};

    // Load data (replace with your dataset)
    load_data("data.csv", &model);

    // Train model
    train(&model, 0.01, 1000);

    // Save results
    save_predictions(&model, "predictions.csv");

    // Free memory
    free(model.X);
    free(model.y);

    printf("Training complete!\n");
    printf("Final equation: y = %fx + %f\n", model.slope, model.intercept);
    return 0;
}