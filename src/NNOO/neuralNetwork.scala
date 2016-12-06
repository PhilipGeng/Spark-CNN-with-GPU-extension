/**
 * Created by philippy on 2015/9/30.
 *
 * constructor:
 *  numOfInput: number of input
 *  numOfOutput: number of output
 *  hiddenLayer: number of neurons in each hidden layer
 *      e.g: Array(2,3): 2 neuron in first hidden layer and 3 neuron in second hidden layer
 *
 */
class neuralNetwork (numOfInput:Int,numOfOutput:Int,hiddenLayer:Array[Int]){
  //initialize layers
  val input:Int = numOfInput
  val output:Int = numOfOutput
  var inputLayer:layer = new layer(input)
  var hidLayer:Array[layer] = hiddenLayer.map{hl=>new layer(hl)}
  var outputLayer:layer = new layer(output,true)//true for output layer
  var layers:Array[layer] = Array(inputLayer)++hidLayer++Array(outputLayer)
  //initialize layer links
  for(i<-0 to layers.length-1){
    if(i!=0)
      layers(i).setPrev(layers(i-1))
    if(i!=layers.length-1)
      layers(i).setNext(layers(i+1))
    //initialize neurons
    layers(i).initNeurons()
  }
  //start forward propagation
  def forwardProp(data:Array[Double]): Array[Double] ={
    inputLayer.forwardProp(data)
  }
  //start backward propagation
  def backProp(target:Array[Double]): Unit ={
    outputLayer.backPropStart(target:Array[Double])
  }
  //data:(data,label)
  def train(data:Array[(Array[Double],Array[Double])]): Unit ={
    data.foreach{data=>
      var fpRes = forwardProp(data._1)
      backProp(data._2)
    }
  }
  def predict(data:Array[Double]): Array[Double] = {
    var res = forwardProp(data)
    res.map(approximate(_))
  }
  def approximate(value:Double): Double = {
    value
  }
}
