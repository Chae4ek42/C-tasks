#include <iostream>
#include "types.hpp"
#include "strategies.hpp"

int main()
{
    Matrix X;
    Vector y_true;

    readCSV("data.csv", X, y_true);
    auto profiledFit = withBenchmarking(fitGradientDescent);

    Vector weights_mse = profiledFit(X, y_true, mseGradientStrategy, 0.01, 5000);
    std::cout << "Weights MSE: ";
    for (size_t i = 0; i < weights_mse.size(); i++)
    {
        std::cout << weights_mse[i] << " ";
    }
    std::cout << std::endl;

    Vector weights_mae = profiledFit(X, y_true, maeGradientStrategy, 0.01, 5000);
    std::cout << "Weights MAE: ";
    for (size_t i = 0; i < weights_mae.size(); i++)
    {
        std::cout << weights_mae[i] << " ";
    }
    std::cout << std::endl;
}