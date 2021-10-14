package com.sidhsuchdev.neuralnetworkjava;

public interface CostFunction {
    double function(double actual, double y);
    double functionDerivative(double actual, double y);
}
