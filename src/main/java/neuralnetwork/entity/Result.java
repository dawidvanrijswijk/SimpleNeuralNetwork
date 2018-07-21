package neuralnetwork.entity;

import neuralnetwork.Analyzer;
import neuralnetwork.parser.result.IResultParser;

public class Result {
    private float successPercentage;
    private float quadraticError;
    private Analyzer analyzer;
    private IResultParser resultParser;

    public Result(Analyzer analyzer, IResultParser resultParser, float successPercentage, float quadraticError) {
        this.analyzer = analyzer;
        this.successPercentage = successPercentage;
        this.quadraticError = quadraticError;
        this.resultParser = resultParser;
    }

    public int predictValue(float[] element) {
        return (Integer) resultParser.parseResult(analyzer.getFOut(element));
    }

    public float getSuccessPercentage() {
        return successPercentage;
    }

    public float getQuadraticError() {
        return quadraticError;
    }
}