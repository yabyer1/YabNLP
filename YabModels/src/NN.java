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
   void newImage(double [][] newImage){
            for(int i = 0; i < newImage.length; i++) {
                for (int j = 0; j < newImage[0].length; j++) {
                    input[i * newImage[0].length + j] = newImage[i][j];

                }
            }
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


}
