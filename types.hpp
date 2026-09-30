#pragma once

#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

using LossGradientStrategy = std::function<Vector(const Vector &y_pred, const Vector &y_true)>;
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