package neuralnetwork.callback;

import neuralnetwork.entity.Errors;
import neuralnetwork.entity.Result;

public interface INeuralNetworkCallback {

    void success(Result result);

    void failure(Errors error);
}
