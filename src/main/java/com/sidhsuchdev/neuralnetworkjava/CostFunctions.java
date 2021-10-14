package com.sidhsuchdev.neuralnetworkjava;

public interface CostFunctions {
    CostFunction QUADRATIC = new QuadraticCostFunction();

    default CostFunction get(String functionName) {
        switch (functionName.toLowerCase()) {
            case "quadratic":
                return QUADRATIC;
            default:
                return null;
        }
    }
}

class QuadraticCostFunction implements CostFunction {
    @Override
    public double function(double actual, double y) {
        return Math.pow(actual - y, 2) / 2;
    }

    @Override
    public double functionDerivative(double actual, double y) {
        return actual - y;
    }
}
