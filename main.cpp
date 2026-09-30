#include <iostream>
#include "types.hpp"
#include "strategies.hpp"

int main()
{
    Matrix X;
    Vector y_true;

    readCSV("data.csv", X, y_true);

    Vector y_pred = {1.0, 1.0, 1.0};

    LossGradientStrategy strategy = mseGradientStrategy;
    Vector grad = strategy(y_pred, y_true);
    for (int i = 0; i < grad.size(); i++)
    {
        std::cout << grad[i] << " ";
    }
    return 0;
}