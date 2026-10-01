#include <iostream>
#include "types.hpp"
#include "strategies.hpp"

int main()
{
    Matrix X;
    Vector y_true;

    readCSV("data.csv", X, y_true);

    Vector weights_mse = fitGradientDescent(X, y_true, mseGradientStrategy);
    std::cout << "Weights MSE: ";
    for (size_t i = 0; i < weights_mse.size(); i++)
    {
        std::cout << weights_mse[i] << " ";
    }
    std::cout << std::endl;

    Vector weights_mae = fitGradientDescent(X, y_true, maeGradientStrategy);
    std::cout << "Weights MAE: ";
    for (size_t i = 0; i < weights_mae.size(); i++)
    {
        std::cout << weights_mae[i] << " ";
    }
    std::cout << std::endl;
}