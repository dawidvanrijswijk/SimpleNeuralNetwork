package neuralnetwork.parser.result;

public interface IResultParser<T> {

    int countSuccesses(int success, float fOut, float t);

    T parseResult(float result);
}
