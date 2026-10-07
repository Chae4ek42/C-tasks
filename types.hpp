#pragma once

#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

using LossGradientStrategy = std::function<Vector(const Vector &, const Vector &)>;
// Произведение векторов
inline double dot(const Vector &a, const Vector &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

// Читает признаки в X и таргеты в y
void readCSV(const std::string &filename, Matrix &X, Vector &y)
{
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::string value;
        Vector row;

        while (std::getline(ss, value, ','))
        {
            row.push_back(std::stod(value));
        }

        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }

    file.close();
}
// Обучение через стратегию
Vector fitGradientDescent(const Matrix &X, const Vector &y, LossGradientStrategy strategy, double lr = 0.01, int epochs = 5000)
{
    size_t num_samples = X.size();
    size_t num_features = X[0].size();

    Vector weights(num_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        Vector y_pred(num_samples, 0.0);
        for (size_t i = 0; i < num_samples; ++i)
        {
            y_pred[i] = dot(X[i], weights);
        }

        Vector loss_grad = strategy(y_pred, y);

        Vector w_grad(num_features, 0.0);
        for (size_t j = 0; j < num_features; ++j)
        {
            for (size_t i = 0; i < num_samples; ++i)
            {
                w_grad[j] += X[i][j] * loss_grad[i];
            }
        }

        for (size_t j = 0; j < num_features; ++j)
        {
            weights[j] -= lr * w_grad[j];
        }
    }

    return weights;
}
// Замер времени
inline auto withBenchmarking(std::function<Vector(const Matrix &, const Vector &, LossGradientStrategy, double, int)> fitFunc)
{
    return [fitFunc](const Matrix &X, const Vector &y, LossGradientStrategy strategy, double lr = 0.01, int epochs = 5000)
    {
        auto start = std::chrono::high_resolution_clock::now();

        Vector result = fitFunc(X, y, strategy, lr, epochs);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Time: " << elapsed.count() << " ms\n";

        return result;
    };
}