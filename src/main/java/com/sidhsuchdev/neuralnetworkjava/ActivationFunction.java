package com.sidhsuchdev.neuralnetworkjava;

public interface ActivationFunction {
    double function(double input);

    double functionDerivative(double input);
}
