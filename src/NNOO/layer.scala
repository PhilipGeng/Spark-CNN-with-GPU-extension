/**
 * Created by philippy on 2015/9/30.
 *
 * constructor:
 * numOfNeuron: number of neurons in this layer
 * prevLayer:previous layer, null for inputlayer
 * nextLayer:next layer, null for output layer
 *
 */
class layer(numOfNeuron:Int,var output:Boolean=false) {
  var prevLayer: layer = null
  var nextLayer: layer = null
  var neurons: Array[neuron] = new Array[neuron](numOfNeuron).map(nr => new neuron())
  //add bias unit for non-output layer
  if (!output)
    neurons ++= Array(new neuron(true))
  var delta: Array[Double] = null

  def initNeurons() {//init neuron links
    if (nextLayer != null)
      neurons.foreach(n => n.initWeight(nextLayer.neurons))
  }

  def setPrev(prev: layer) {//init previous layer
    prevLayer = prev
  }

  def setNext(next: layer) {//init next layer
    nextLayer = next
  }

  def forwardProp(data: Array[Double]): Array[Double] = {
    if (numOfNeuron != data.length)//bad data
      null
    else{
      if(output) {//direct return for output layer
        neurons.filter(n => n.bias == false).zip(data).foreach(n => n._1.output = n._2)
        data
      }
      else {
        //set neuron output value for this layer
        neurons.filter(n => n.bias == false).zip(data).foreach(n => n._1.output = n._2)
        //calculate neuron output value for next layer and call next layer for fp
        nextLayer.forwardProp(neurons.map(n => n.calValue()).reduceLeft((a, b) => a.zip(b).map(x => x._1 + x._2)).map(sigmoid(_)))
      }
    }
  }

  def backPropStart(target: Array[Double]): Unit = {
    delta = neurons.zip(target).map(x => x._1.output * (1.0 - x._1.output) * (x._2 - x._1.output))
    prevLayer.backProp(delta)
  }

  def backProp(prevDelta: Array[Double]): Unit = {
    adjustWeight(prevDelta)
    if (prevLayer != null) {
      delta = neurons.filter(n => n.bias == false).map(n => n.calDelta(prevDelta))
      prevLayer.backProp(delta)
    }
  }

  def adjustWeight(prevDelta: Array[Double]): Unit = {
    neurons.foreach(x => x.adjustWeight(prevDelta))
  }

  def sigmoid(input:Double): Double = 1.0 / (1.0 + Math.exp(-input))

}