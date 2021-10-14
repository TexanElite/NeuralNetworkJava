package com.sidhsuchdev.neuralnetworkjava;

public interface ActivationFunctions {
    ActivationFunction SIGMOID = new SigmoidActivationFunction();
    ActivationFunction RELU = new ReLUActivationFunction();

    default ActivationFunction get(String functionName) {
        switch (functionName.toLowerCase()) {
            case "sigmoid":
                return SIGMOID;
            case "relu":
                return RELU;
            default:
                return null;
        }
    }
}

class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public double function(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double functionDerivative(double input) {
        return function(input) * (1 - function(input));
    }
}


class ReLUActivationFunction implements ActivationFunction {
    @Override
    public double function(double input) {
        return Math.max(0, input);
    }

    @Override
    public double functionDerivative(double input) {
        if (input < 0) return 0;
        return 1;
    }
}
