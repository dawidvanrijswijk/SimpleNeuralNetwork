package example;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.callback.INeuralNetworkCallback;
import neuralnetwork.entity.Errors;
import neuralnetwork.entity.Result;
import neuralnetwork.utils.DataUtils;

public class Main {
    public static void main(String[] args) {
        System.out.println("Starting neural network sample... ");

        float[][] x = DataUtils.readInputsFromFile("/Users/dawidvanrijswijk/Desktop/data/x.txt");
        int[] t = DataUtils.readOutputsFromFile("/Users/dawidvanrijswijk/Desktop/data/t.txt");

        NeuralNetwork neuralNetwork = new NeuralNetwork(x, t, new INeuralNetworkCallback() {
            @Override
            public void success(Result result) {
                float[] valueToPredict = new float[]{-0.205f, 0.780f};
                System.out.println("Success percentage: " + result.getSuccessPercentage());
                System.out.println("Predicted result: " + result.predictValue(valueToPredict));
            }


            @Override
            public void failure(Errors error) {
                System.out.println("Error: " + error.getDescription());
            }
        });

        neuralNetwork.startLearning();
    }
}