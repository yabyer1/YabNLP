import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NN{
     double [] input;
    List<double []> hiddenLayers = new ArrayList<>();
    double [] inputWeights;
    List<double []> bias = new ArrayList<>();
    List<double [][]> hiddenLayerWeights = new ArrayList<>();
    double [] output = new double[10]; // we will be classifying numbers 0 - 9
    double [][] outputWeights;
    double[] outputBias = new double[10];
    double [] expected = new double [10];
    public NN(int inputSize, int numHiddenLayers, int dimensionHiddenLayers){
        input = new double[inputSize];
        inputWeights = new double[inputSize];
        for(int i = 0; i < inputSize; i++){
            inputWeights[i] = Math.random();
        }
                for(int i = 0; i < numHiddenLayers; i++){
                    hiddenLayers.add(new double[dimensionHiddenLayers]);
                    bias.add(new double[dimensionHiddenLayers]);
                    hiddenLayerWeights.add(new double[dimensionHiddenLayers][i == 0 ? input.length : dimensionHiddenLayers]);
                    for(int j = 0; j < dimensionHiddenLayers; j++){
                        bias.get(i)[j] =   Math.random();

                        for(double [] p : hiddenLayerWeights.getLast()){
                            Arrays.fill(p, Math.random());
                        }
                    }
                    
                }
                double [][] outputWeights = new double[10][dimensionHiddenLayers];
                double [] outputBias = new double[10];
                for(int i = 0; i < 10; i++){
                    outputBias[i] = Math.random();
                    Arrays.fill(outputWeights[i], Math.random());
                }

    }
   void newImage(double [][] newImage, int lbl){
            for(int i = 0; i < newImage.length; i++) {
                for (int j = 0; j < newImage[0].length; j++) {
                    input[i * newImage[0].length + j] = newImage[i][j];

                }
            }
            expected[lbl] = 1;
   }
    
    
    public void forward(){ 

      for(int i = 0; i < hiddenLayers.getFirst().length; i++){
          double sum = 0;
          for(int j = 0; j < input.length; j++){
              sum += (inputWeights[j] * input[j]);

          }
          sum += bias.getFirst()[i];
          hiddenLayers.getFirst()[i] = RELU(sum);
      }
      for(int i = 1; i < hiddenLayers.size(); i++){
          double [] a = hiddenLayers.get(i - 1);
          for(int j = 0; j < hiddenLayers.get(i).length; j++){
              double sum = bias.get(i)[j];
              for(int k = 0; k < a.length; k++){
                        sum += (hiddenLayerWeights.get(i)[j][k] * a[k]);
              }
              hiddenLayers.get(i)[j] = RELU(sum);
          }
      }
        double [] a = hiddenLayers.getLast();
      for(int i = 0; i < output.length; i++){

          double sum = outputBias[i];
          for(int j = 0; j < a.length; j++){
              sum += (outputWeights[i][j] * a[j]);
          }
          output[i] = sum;
      }
      softmax(output);
    }

    private void softmax(double[] output) {
        double sum = 0;
        for (double v : output) {
            sum += Math.pow(Math.E, v);
        }
        for(int i = 0; i < output.length; i++){
            output[i] = Math.pow(Math.E, output[i]) / sum;
        }

    }

    private double RELU(double v) {
        return Math.max(0, v);
    }
    private void backward(double learningrate){
      double [] costGradient = new double[10];
      for(int i = 0; i < costGradient.length; i++){
          costGradient[i] = output[i] - expected[i];
      }
       double [] gradient  = new double[hiddenLayers.getLast().length];
       for(int i = 0; i < gradient.length; i++){

           double reluDiv = (hiddenLayers.getLast()[i] > 0) ? 1 : 0;
           gradient[i] = 0;
           for(int j = 0; j < 10; j++){

                gradient[i] += (costGradient[j] * outputWeights[j][i]) * reluDiv;
           }

       }
       double[] prevGrads;
       prevGrads = gradient;
       double [] prevActivations = hiddenLayers.getLast(); // last hidden layer needed to step update output layer
       for(int i = 0; i < 10; i++){
           for(int j = 0; j < prevActivations.length; j++){
               outputWeights[i][j] = outputWeights[i][j] - (learningrate * costGradient[i] * prevActivations[j]);
           }
           outputBias[i] -= learningrate * costGradient[i];
       }
       double [] layerbeforelast = (hiddenLayers.size() > 1) ? hiddenLayers.get(hiddenLayers.size() - 2) : input;

       for(int i = 0; i < prevActivations.length; i++){
           for(int j = 0; j < layerbeforelast.length; j++){
               hiddenLayerWeights.getLast()[i][j] = hiddenLayerWeights.getLast()[i][j] -  ( learningrate * gradient[i] * layerbeforelast[j]);
           }
           bias.getLast()[i] = bias.getLast()[i] - (learningrate * gradient[i]);
       }
            for(int i = hiddenLayers.size() - 2; i > -1; i--){
                double [] curLayerGradient = new double[hiddenLayers.get(i).length];
                for(int j = 0; j < curLayerGradient.length; j++){
                    double reluDiv = (hiddenLayers.get(i)[j] > 0) ? 1 : 0;
                    curLayerGradient[j] = 0;
                    for(int k = 0; k < hiddenLayers.get(i + 1).length; k++){
                        curLayerGradient[j] += (prevGrads[k] * hiddenLayerWeights.get(i + 1)[k][j]) * reluDiv;
                    }
                }
                prevGrads = curLayerGradient;
                double [] lbl = (i > 0) ? hiddenLayers.get(i - 1) : input;
                for(int j = 0; j < curLayerGradient.length; j++){
                    for(int k = 0; k < lbl.length; k++){
                        hiddenLayerWeights.get(i)[j][k] = hiddenLayerWeights.get(i)[j][k] - (learningrate * curLayerGradient[j] * lbl[k]);
                    }
                    bias.get(i)[j] = bias.get(i)[j] - (learningrate * curLayerGradient[j]);
                }
            }



    }

}
