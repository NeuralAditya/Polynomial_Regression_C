#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    double *X, *y;
    int n_samples;
    int degree;
    double *coefficients;
} PolynomialModel;

// Function prototypes
void load_data(const char *filename, PolynomialModel *model);
void train(PolynomialModel *model);
double predict(PolynomialModel *model, double x);
void save_predictions(PolynomialModel *model, const char *filename);
void free_model(PolynomialModel *model);

// Load dataset from CSV file
void load_data(const char *filename, PolynomialModel *model) {
    char path[256];
    snprintf(path, sizeof(path), "../data/%s", filename);
    
    FILE *file = fopen(path, "r");
    if (!file) {
        printf("Error: Unable to open file at %s\n", path);
        perror("Details");
        printf("Ensure the file exists in the 'data' directory.\n");
        exit(1);
    }

    int n = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) if (ch == '\n') n++;
    rewind(file);

    if (n == 0) {
        printf("Error: Empty dataset in %s\n", path);
        exit(1);
    }

    model->X = malloc(n * sizeof(double));
    model->y = malloc(n * sizeof(double));
    model->n_samples = n;

    char header[256];
    fgets(header, sizeof(header), file);

    for (int i = 0; i < n; i++) {
        if (fscanf(file, "%lf,%lf\n", &model->X[i], &model->y[i]) != 2) {
            printf("Error: Invalid data format in line %d\n", i + 2);
            exit(1);
        }
    }
    fclose(file);
}

// Train model using Normal Equation: θ = (X^T * X)^(-1) * X^T * Y
void train(PolynomialModel *model) {
    int n = model->n_samples;
    int d = model->degree;
    int cols = d + 1;

    double **X_matrix = malloc(n * sizeof(double *));
    double *Y_vector = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        X_matrix[i] = malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            X_matrix[i][j] = pow(model->X[i], j);
        }
        Y_vector[i] = model->y[i];
    }

    double XT_X[cols][cols], XT_Y[cols];
    memset(XT_X, 0, sizeof(XT_X));
    memset(XT_Y, 0, sizeof(XT_Y));

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < n; k++) {
                XT_X[i][j] += X_matrix[k][i] * X_matrix[k][j];
            }
        }
        for (int k = 0; k < n; k++) {
            XT_Y[i] += X_matrix[k][i] * Y_vector[k];
        }
    }

    double theta[cols];
    for (int i = 0; i < cols; i++) theta[i] = XT_Y[i];

    for (int i = 0; i < cols; i++) {
        if (fabs(XT_X[i][i]) < 1e-9) {
            for (int k = i + 1; k < cols; k++) {
                if (fabs(XT_X[k][i]) > 1e-9) {
                    for (int j = 0; j < cols; j++) {
                        double temp = XT_X[i][j];
                        XT_X[i][j] = XT_X[k][j];
                        XT_X[k][j] = temp;
                    }
                    double temp = theta[i];
                    theta[i] = theta[k];
                    theta[k] = temp;
                    break;
                }
            }
        }

        double diag = XT_X[i][i];
        for (int j = 0; j < cols; j++) XT_X[i][j] /= diag;
        theta[i] /= diag;

        for (int k = 0; k < cols; k++) {
            if (i != k) {
                double factor = XT_X[k][i];
                for (int j = 0; j < cols; j++) XT_X[k][j] -= factor * XT_X[i][j];
                theta[k] -= factor * theta[i];
            }
        }
    }

    model->coefficients = malloc(cols * sizeof(double));
    for (int i = 0; i < cols; i++) {
        model->coefficients[i] = theta[i];
    }

    for (int i = 0; i < n; i++) free(X_matrix[i]);
    free(X_matrix);
    free(Y_vector);
}

// Predict output for a given x
double predict(PolynomialModel *model, double x) {
    double result = 0;
    for (int i = 0; i <= model->degree; i++) {
        result += model->coefficients[i] * pow(x, i);
    }
    return result;
}

// Save predictions to CSV
void save_predictions(PolynomialModel *model, const char *filename) {
    char path[256];
    snprintf(path, sizeof(path), "../data/%s", filename);
    
    FILE *file = fopen(path, "w");
    if (!file) {
        printf("Error: Unable to create %s\n", path);
        exit(1);
    }
    
    fprintf(file, "x,y_true,y_pred\n");
    for (int i = 0; i < model->n_samples; i++) {
        fprintf(file, "%f,%f,%f\n", model->X[i], model->y[i], predict(model, model->X[i]));
    }
    fclose(file);
}

// Free allocated memory
void free_model(PolynomialModel *model) {
    free(model->X);
    free(model->y);
    free(model->coefficients);
}

// Main function
int main() {
    PolynomialModel model = {0};
    model.degree = 2;  // Change degree as needed

    printf("Checking dataset...\n");
    load_data("synthetic.csv", &model);
    
    printf("Training model...\n");
    train(&model);
    
    printf("Saving predictions...\n");
    save_predictions(&model, "predictions.csv");

    printf("\nPolynomial coefficients:\n");
    for (int i = 0; i <= model.degree; i++) {
        printf("Theta_%d = %.4f\n", i, model.coefficients[i]); // θ symbols don't work in Windows Terminal
    }

    free_model(&model);

    printf("\nPress Enter to exit...\n");
    getchar();  // Keeps terminal open
    return 0;
}
