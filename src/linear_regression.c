#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    double *X, *y;
    int n_samples;
    double slope, intercept;
} LinearModel;

// Function prototypes
void load_data(const char *filename, LinearModel *model);
double compute_loss(LinearModel *model);
void train(LinearModel *model, double lr, int epochs);
void save_predictions(LinearModel *model, const char *filename);

void load_data(const char *filename, LinearModel *model) {
    char path[256];
    snprintf(path, sizeof(path), "data/%s", filename);
    
    FILE *file = fopen(path, "r");
    if (!file) {
        printf("Error opening: %s\n", path);
        perror("Details");
        exit(1);
    }

    int n = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) if (ch == '\n') n++;
    rewind(file);

    model->X = malloc(n * sizeof(double));
    model->y = malloc(n * sizeof(double));
    model->n_samples = n;

    char header[256];
    fgets(header, sizeof(header), file);

    for (int i = 0; i < n; i++) {
        if (fscanf(file, "%lf,%lf\n", &model->X[i], &model->y[i]) != 2) {
            printf("Error reading line %d\n", i+2);
            exit(1);
        }
    }
    fclose(file);
}

double compute_loss(LinearModel *model) {
    double loss = 0;
    for (int i = 0; i < model->n_samples; i++) {
        double pred = model->slope * model->X[i] + model->intercept;
        loss += pow(pred - model->y[i], 2);
    }
    return loss / model->n_samples;
}

void train(LinearModel *model, double lr, int epochs) {
    for (int e = 0; e < epochs; e++) {
        double grad_slope = 0, grad_intercept = 0;
        for (int i = 0; i < model->n_samples; i++) {
            double pred = model->slope * model->X[i] + model->intercept;
            grad_slope += (pred - model->y[i]) * model->X[i];
            grad_intercept += (pred - model->y[i]);
        }
        model->slope -= lr * (grad_slope / model->n_samples);
        model->intercept -= lr * (grad_intercept / model->n_samples);
        if (e % 100 == 0) 
            printf("Epoch %d, Loss: %f\n", e, compute_loss(model));
    }
}

void save_predictions(LinearModel *model, const char *filename) {
    char path[256];
    snprintf(path, sizeof(path), "data/%s", filename);
    
    FILE *file = fopen(path, "w");
    if (!file) {
        perror("Error creating predictions file");
        exit(1);
    }
    
    fprintf(file, "x,y_true,y_pred\n");
    for (int i = 0; i < model->n_samples; i++) {
        fprintf(file, "%f,%f,%f\n", model->X[i], model->y[i], 
                model->slope * model->X[i] + model->intercept);
    }
    fclose(file);
}

int main() {
    LinearModel model = {0};
    load_data("data.csv", &model);
    train(&model, 0.01, 1000);
    save_predictions(&model, "predictions.csv");
    free(model.X);
    free(model.y);
    printf("Final model: y = %.4fx + %.4f\n", model.slope, model.intercept);
    return 0;
}