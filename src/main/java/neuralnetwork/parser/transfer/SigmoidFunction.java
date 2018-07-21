package neuralnetwork.parser.transfer;

public class SigmoidFunction implements ITransferFunction {
    @Override
    public float transfer(float value) {
        return (float)(1/(1+Math.exp(-value)));
    }
}
