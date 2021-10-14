package com.sidhsuchdev.neuralnetworkjava;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] values;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.values = new double[rows][cols];
    }

    public Matrix(double[] vector) {
        this(vector.length, 1);
        for (int i = 0; i < vector.length; i++) {
            values[i][0] = vector[i];
        }
    }

    public Matrix(double[][] values) {
        this.rows = values.length;
        this.cols = values[0].length;
        this.values = values;
    }

    public Matrix add(Matrix other) {
        if (rows != other.rows && cols != other.cols) {
            throw new IllegalArgumentException("Invalid Matrix: Rows and columns of operands are not equal");
        }
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = values[i][j] + other.values[i][j];
            }
        }
        return new Matrix(newValues);
    }

    public Matrix sub(Matrix other) {
        if (rows != other.rows && cols != other.cols) {
            throw new IllegalArgumentException("Invalid Matrix: Rows and columns of operands are not equal");
        }
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = values[i][j] - other.values[i][j];
            }
        }
        return new Matrix(newValues);
    }

    public Matrix multiply(Matrix other) {
        if (cols != other.rows) {
            throw new IllegalArgumentException("Invalid Matrix: Shapes are not compatible to multiply");
        }
        double[][] newValues = new double[rows][other.cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    newValues[i][j] += values[i][k] * other.values[k][j];
                }
            }
        }
        return new Matrix(newValues);
    }

    public Matrix multiply(double scalar) {
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = values[i][j] * scalar;
            }
        }
        return new Matrix(newValues);
    }

    public Matrix hadamardProduct(Matrix other) {
        if (rows != other.rows && cols != other.cols) {
            throw new IllegalArgumentException("Invalid Matrix: Rows and columns of operands are not equal");
        }
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = values[i][j] * other.values[i][j];
            }
        }
        return new Matrix(newValues);
    }

    public Matrix transpose() {
        double[][] newValues = new double[cols][rows];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                newValues[i][j] = values[j][i];
            }
        }
        return new Matrix(newValues);
    }

    public Matrix applyFunction(UnaryOperator<Double> function) {
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = function.apply(values[i][j]);
            }
        }
        return new Matrix(newValues);
    }

    public Matrix applyFunction(Matrix other, BinaryOperator<Double> function) {
        if (rows != other.rows && cols != other.cols) {
            throw new IllegalArgumentException("Invalid Matrix: Rows and columns of operands are not equal");
        }
        double[][] newValues = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newValues[i][j] = function.apply(values[i][j], other.values[i][j]);
            }
        }
        return new Matrix(newValues);
    }

    public double[] vectorize() {
        if (cols != 1) {
            throw new IllegalArgumentException("Invalid Matrix: Columns must be 1");
        }
        double[] vector = new double[rows];
        for (int i = 0; i < rows; i++) {
            vector[i] = values[i][0];
        }
        return vector;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] getValues() {
        return values;
    }

    public double get(int row, int col) {
        return values[row][col];
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Matrix)) return false;
        Matrix otherMatrix = (Matrix) o;
        if (rows != otherMatrix.rows || cols != otherMatrix.cols) return false;
        return Arrays.deepEquals(values, otherMatrix.values);
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder("[\n");
        for (double[] v : values) {
            res.append("\t");
            res.append(Arrays.toString(v));
            res.append("\n");
        }
        res.append("]");
        return res.toString();
    }
}