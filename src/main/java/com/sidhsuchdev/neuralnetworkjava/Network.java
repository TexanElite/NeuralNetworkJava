package com.sidhsuchdev.neuralnetworkjava;

import java.util.*;

public class Network {

    private int layers;

    private List<Matrix> weightLayers;
    private List<Matrix> biasLayers;
    private List<ActivationFunction> activationFunctionLayers;
    private int inputSize;
    private CostFunction costFunction;

    public Network(int inputSize, CostFunction costFunction) {
        this.inputSize = inputSize;
        this.costFunction = costFunction;

        this.layers = 1;
        this.weightLayers = new ArrayList<>();
        this.biasLayers = new ArrayList<>();
        this.activationFunctionLayers = new ArrayList<>();
    }

    public void addLayer(int size, ActivationFunction activationFunction) {
        int previousSize;
        if (layers++ == 1) previousSize = inputSize;
        else previousSize = biasLayers.get(biasLayers.size() - 1).getRows();
        weightLayers.add(new Matrix(size, previousSize));
        biasLayers.add(new Matrix(size, 1));
        activationFunctionLayers.add(activationFunction);

    }

    public void randomizeWeightsAndBiases() {
        for (int i = 0; i < layers - 1; i++) {
            biasLayers.set(i, biasLayers.get(i).applyFunction(n -> randomNormal()));
            weightLayers.set(i, weightLayers.get(i).applyFunction(n -> randomNormal()));
        }
    }

    public void train(Matrix[][] trainingData, int epochs, int batchSize, double learningRate) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            trainingData = shuffleDataRows(trainingData);
            Matrix[][][] batches = createBatches(trainingData, batchSize);
            for (Matrix[][] batch : batches) {
                updateBatch(batch, learningRate);
            }
        }
    }

    public void updateBatch(Matrix[][] batch, double learningRate) {
        Matrix[] nablaB = new Matrix[biasLayers.size()];
        Matrix[] nablaW = new Matrix[weightLayers.size()];

        for (int i = 0; i < layers - 1; i++) {
            nablaB[i] = new Matrix(biasLayers.get(i).getRows(), biasLayers.get(i).getCols());
            nablaW[i] = new Matrix(weightLayers.get(i).getRows(), weightLayers.get(i).getCols());
        }

        for (Matrix[] matrices : batch) {
            Map.Entry<Matrix[], Matrix[]> backpropagationResults = backpropagation(matrices[0], matrices[1]);
            Matrix[] deltaNablaB = backpropagationResults.getKey();
            Matrix[] deltaNablaW = backpropagationResults.getValue();
            for (int j = 0; j < layers - 1; j++) {

                nablaB[j] = nablaB[j].add(deltaNablaB[j]);
                nablaW[j] = nablaW[j].add(deltaNablaW[j]);
            }
        }
        for (int i = 0; i < layers - 1; i++) {
            biasLayers.set(i, biasLayers.get(i).sub(nablaB[i].multiply(learningRate / batch.length)));
            weightLayers.set(i, weightLayers.get(i).sub(nablaW[i].multiply(learningRate / batch.length)));
        }
    }

    public Matrix feedForward(Matrix input) {
        Matrix current = input;
        for (int i = 0; i < layers - 1; i++) {
            current = weightLayers.get(i).multiply(current);
            current = current.add(biasLayers.get(i));
            current = current.applyFunction(activationFunctionLayers.get(i)::function);
        }
        return current;
    }

    public Map.Entry<Matrix[], Matrix[]> backpropagation(Matrix x, Matrix y) {
        Matrix[] nablaB = new Matrix[biasLayers.size()];
        Matrix[] nablaW = new Matrix[weightLayers.size()];

        Matrix curActivation = x;
        Matrix[] activations = new Matrix[layers];
        activations[0] = curActivation;
        Matrix[] zValues = new Matrix[layers];

        for (int i = 0; i < layers - 1; i++) {
            Matrix curWeights = weightLayers.get(i);
            Matrix curBiases = biasLayers.get(i);
            Matrix curZ = curWeights.multiply(curActivation).add(curBiases);
            zValues[i + 1] = curZ;
            curActivation = curZ.applyFunction(activationFunctionLayers.get(i)::function);
            activations[i + 1] = curActivation;
        }

        Matrix delta = curActivation.applyFunction(y, costFunction::functionDerivative)
                .hadamardProduct(zValues[layers - 1].applyFunction(activationFunctionLayers.get(activationFunctionLayers.size() - 1)::functionDerivative));

        nablaB[nablaB.length - 1] = delta;
        nablaW[nablaW.length - 1] = delta.multiply(activations[activations.length - 2].transpose());

        for (int l = 2; l < layers; l++) {
            Matrix curZ = zValues[layers - l];
            delta = weightLayers.get(layers - l).transpose().multiply(delta)
                    .hadamardProduct(curZ.applyFunction(activationFunctionLayers.get(activationFunctionLayers.size() - l)::functionDerivative));
            nablaB[nablaB.length - l] = delta;
            nablaW[nablaW.length - l] = delta.multiply(activations[layers - l - 1].transpose());
        }
        return new AbstractMap.SimpleEntry<>(nablaB, nablaW);
    }

    private static Matrix[][] shuffleDataRows(Matrix[][] data) {
        int length = data.length;
        ArrayList<Integer> shifts = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            shifts.add(i);
        }
        Collections.shuffle(shifts);
        Matrix[][] shuffledData = new Matrix[length][2];
        for (int i = 0; i < length; i++) {
            shuffledData[i] = data[shifts.get(i)];
        }
        return shuffledData;
    }

    private Matrix[][][] createBatches(Matrix[][] data, int batchSize) {
        Matrix[][][] batches = new Matrix[(data.length + batchSize - 1) / batchSize][][];
        Matrix[][] curBatch = null;

        for (int i = 0; i < data.length; i++) {
            int curBatchIndex = i / batchSize;
            if (i % batchSize == 0) {
                int size = Math.min(data.length - i, batchSize);
                curBatch = new Matrix[size][];
            }
            curBatch[i % batchSize] = data[i];
            if (i % batchSize == batchSize - 1 || i == data.length - 1) {
                batches[curBatchIndex] = curBatch;
            }
        }
        return batches;
    }

    public int argmax(Matrix data) {
        double[] values = data.vectorize();
        double bestValue = values[0];
        int bestIndex = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private double randomNormal() {
        double u = Math.random();
        double v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

}