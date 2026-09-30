#pragma once

#include "types.hpp"

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