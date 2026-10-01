#pragma once

#include "types.hpp"
double sign(double val)
{
    if (val > 0)
        return 1.0;
    if (val < 0)
        return -1.0;
    return 0.0;
}
// Mse лосс
Vector mseGradientStrategy(const Vector &y_pred, const Vector &y_true)
{
    size_t n = y_pred.size();
    Vector grad(n);

    for (size_t i = 0; i < n; i++)
    {
        grad[i] = (2.0 / n) * (y_pred[i] - y_true[i]);
    }

    return grad;
}
// Mae лосс
Vector maeGradientStrategy(const Vector &y_pred, const Vector &y_true)
{
    size_t n = y_pred.size();
    Vector grad(n);

    for (size_t i = 0; i < n; i++)
    {
        grad[i] = (1.0 / n) * sign(y_pred[i] - y_true[i]);
    }

    return grad;
}