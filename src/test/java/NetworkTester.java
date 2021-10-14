import com.sidhsuchdev.neuralnetworkjava.ActivationFunctions;
import com.sidhsuchdev.neuralnetworkjava.CostFunctions;
import com.sidhsuchdev.neuralnetworkjava.Matrix;
import com.sidhsuchdev.neuralnetworkjava.Network;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class NetworkTester {

    @Test
    public void matrixTest() {
        Matrix fib = new Matrix(new double[][] {{1, 1}, {1, 0}} );
        Matrix vec = new Matrix(new double[] {1, 0} );
        Assertions.assertEquals(fib.multiply(vec), new Matrix(new double[] {1, 1}));
        Assertions.assertEquals(fib.add(fib), new Matrix(new double[][] {{2, 2}, {2, 0}}));
        Assertions.assertEquals(fib.hadamardProduct(fib), new Matrix(new double[][] {{1, 1}, {1, 0}}));
        Assertions.assertEquals(fib.sub(fib), new Matrix(new double[][] {{0, 0}, {0, 0}}));
        Assertions.assertEquals(fib.multiply(fib), new Matrix(new double[][] {{2, 1}, {1, 1}}));
        Assertions.assertEquals(fib.multiply(3), new Matrix(new double[][] {{3, 3}, {3, 0}}));
    }

    @Test
    public void basicNetworkTest1() {
        Network network = new Network(1, CostFunctions.QUADRATIC);
        network.addLayer(30, ActivationFunctions.SIGMOID);
        network.addLayer(2, ActivationFunctions.SIGMOID);
        Matrix[][] trainingData = new Matrix[10000][2];
        Matrix[][] testingData = new Matrix[1000][2];
        network.randomizeWeightsAndBiases();
        for (int i = 0; i < trainingData.length; i++) {
            Matrix x = new Matrix(new double[] {Math.random() * 2});
            Matrix y = x.get(0, 0) > 1 ? new Matrix(new double[] {0, 1}) : new Matrix(new double[] {1, 0});
            trainingData[i][0] = x;
            trainingData[i][1] = y;
        }
        for (int i = 0; i < testingData.length; i++) {
            Matrix x = new Matrix(new double[] {Math.random() * 2});
            Matrix y = x.get(0, 0) > 1 ? new Matrix(new double[] {0, 1}) : new Matrix(new double[] {1, 0});
            testingData[i][0] = x;
            testingData[i][1] = y;
        }
        network.train(trainingData, 30, 10, 0.1);
        int count = 0;
        for (Matrix[] test : testingData) {
            int actual = network.argmax(test[1]);
            int result = network.argmax(network.feedForward(test[0]));
            if (actual == result) count++;
        }
        Assertions.assertTrue((double) count / testingData.length >= .90);
    }

    @Test
    public void basicNetworkTest2() {
        Network network = new Network(2, CostFunctions.QUADRATIC);
        network.addLayer(30, ActivationFunctions.SIGMOID);
        network.addLayer(2, ActivationFunctions.SIGMOID);
        Matrix[][] trainingData = new Matrix[10000][2];
        Matrix[][] testingData = new Matrix[1000][2];
        network.randomizeWeightsAndBiases();
        for (int i = 0; i < trainingData.length; i++) {
            Matrix x = new Matrix(new double[] {Math.random() * 2, Math.random() * 2});
            Matrix y = x.get(0, 0) + x.get(1, 0) > 2 ? new Matrix(new double[] {0, 1}) : new Matrix(new double[] {1, 0});
            trainingData[i][0] = x;
            trainingData[i][1] = y;
        }
        for (int i = 0; i < testingData.length; i++) {
            Matrix x = new Matrix(new double[] {Math.random() * 2, Math.random() * 2});
            Matrix y = x.get(0, 0) + x.get(1, 0) > 2 ? new Matrix(new double[] {0, 1}) : new Matrix(new double[] {1, 0});
            testingData[i][0] = x;
            testingData[i][1] = y;
        }
        network.train(trainingData, 30, 10, 0.1);
        int count = 0;
        for (Matrix[] test : testingData) {
            int actual = network.argmax(test[1]);
            int result = network.argmax(network.feedForward(test[0]));
            if (actual == result) count++;
        }
        Assertions.assertTrue((double) count / testingData.length >= .90);
    }

}
