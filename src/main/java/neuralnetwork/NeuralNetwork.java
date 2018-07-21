package neuralnetwork;

import neuralnetwork.callback.INeuralNetworkCallback;
import neuralnetwork.entity.Result;
import neuralnetwork.exception.NotSameInputOutputSizeException;
import neuralnetwork.exception.ZeroInputDimensionException;
import neuralnetwork.exception.ZeroInputElementsException;
import neuralnetwork.exception.ZeroNeuronsException;
import neuralnetwork.parser.result.BinaryResultParser;
import neuralnetwork.parser.result.IResultParser;
import neuralnetwork.parser.transfer.ITransferFunction;
import neuralnetwork.parser.transfer.SigmoidFunction;
import neuralnetwork.utils.Utils;

import static neuralnetwork.entity.Errors.*;

public class NeuralNetwork {
    private int nElements;
    private int dimension;
    private float[][] inputs;
    private int[] outputs;
    private float[] bias;
    private float[] vWeights;
    private float[][] wWeights;
    private int neurons;
    private float bOut;
    // Default iterations limit
    private int iterationsLimit = 10000;

    private Analyzer analyzer;
    private IResultParser resultParser;
    private ITransferFunction transferFunction;

    private INeuralNetworkCallback neuralNetworkCallback = null;

    public NeuralNetwork (float[][] inputs, int[] output, INeuralNetworkCallback neuralNetworkCallback) {
        bOut = Utils.randFloat(-0.5f, 0.5f);
        this.neuralNetworkCallback = neuralNetworkCallback;
        // Default transfer function
        this.transferFunction = new SigmoidFunction();
        // Default result parser
        this.resultParser = new BinaryResultParser();

        this.inputs = inputs;
        this.outputs = output;

        this.nElements = output.length;
        try {
            this.dimension = inputs[0].length;
        } catch (ArrayIndexOutOfBoundsException e){
            neuralNetworkCallback.failure(NOT_SAME_INPUT_OUTPUT);
        }

        // Default num neurons = dimension
        this.neurons = dimension;
    }

    public void startLearning(){
        try {
            if (inputs.length != outputs.length)
                throw new NotSameInputOutputSizeException();
            if (inputs.length == 0)
                throw new ZeroInputElementsException();

            HiddenLayerNeuron hiddenLayerNeuron = new HiddenLayerNeuron (neurons, dimension);
            bias = hiddenLayerNeuron.getBias();
            vWeights = hiddenLayerNeuron.getVWeights();
            wWeights = hiddenLayerNeuron.getWWeights();

            new NeuralNetworkThread().run();

        } catch (NotSameInputOutputSizeException e) {
            neuralNetworkCallback.failure(NOT_SAME_INPUT_OUTPUT);
        } catch (ZeroInputDimensionException e) {
            neuralNetworkCallback.failure(ZERO_INPUT_DIMENSION);
        } catch (ZeroInputElementsException e) {
            neuralNetworkCallback.failure(ZERO_INPUT_ELEMENTS);
        } catch (ZeroNeuronsException e) {
            neuralNetworkCallback.failure(ZERO_NEURONS);
        }
    }

    private float[] getRowElements(int row){
        float[] elements = new float[dimension];
        for (int i = 0; i<dimension; i++){
            elements[i] = this.inputs[row][i];
        }
        return elements;
    }

    public void setTransferFunction(ITransferFunction transferFunction){
        this.transferFunction = transferFunction;
    }

    public int getNeurons() {
        return neurons;
    }

    public void setNeurons(int neurons) {
        this.neurons = neurons;
    }

    public void setResultParser(IResultParser resultParser) {
        this.resultParser = resultParser;
    }

    public int getIterationsLimit() {
        return iterationsLimit;
    }

    public void setIterationsLimit(int iterationsLimit) {
        this.iterationsLimit = iterationsLimit;
    }

    public class NeuralNetworkThread implements Runnable {
        @Override
        public void run() {
            float quadraticError = 0;
            float[] f;
            int success = 0;
            for (int i = 0; i<iterationsLimit; i++) {
                success = 0;
                for (int z = 0; z<nElements; z++) {
                    analyzer = new Analyzer (getRowElements(z), wWeights, bias, vWeights, bOut, neurons, transferFunction, dimension);
                    f = analyzer.getFOutArray();
                    float fOut = analyzer.getFOut();
                    Learner learner = new Learner(outputs[z], fOut, f, vWeights, wWeights, bias, bOut, neurons, getRowElements(z), dimension);
                    vWeights = learner.getVWeights();
                    wWeights = learner.getWWeights();
                    bias = learner.getBias();
                    bOut = learner.getBOut();
                    success = resultParser.countSuccesses(success, fOut, outputs[z]);
                    quadraticError += Math.pow(((outputs[z] - fOut)), 2);
                }
                quadraticError *= 0.5f;
            }
            float successPercentage = (success / (float)nElements) * 100;
            Result result = new Result(analyzer, resultParser, successPercentage, quadraticError);
            neuralNetworkCallback.success(result);
        }
    }
}
