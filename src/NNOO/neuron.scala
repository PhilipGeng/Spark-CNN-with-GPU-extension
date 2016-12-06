/**
 * Created by philippy on 2015/9/30.
 * 
 */
class neuron(var bias:Boolean = false, var eta:Double = 0.1, var momentum:Double = 0, var initWeight:Double = 0d) {
  var weight:Array[(neuron,Double)] = null
  var delta: Double = 0
  var prevWeight: Array[(neuron,Double)] = weight
  var output:Double = 0
  //output=1 for bias units
  if(bias)
    output = 1
  //init sigout with [(next neuron,weight)]
  def initWeight(out:Array[neuron]): Unit ={
    //init sigout with [(next neuron, weight)] except the bias unit
    weight = out.map((_,initWeight)).filter(x=>x._1.bias==false)
    //init prevSigOut with weight 0
    prevWeight = out.map((_,0.0)).filter(x=>x._1.bias==false)
  }
  //calculate forward propagation value vector to next layer
  def calValue(): Array[Double] = {
    weight.map(x=>x._2*output)
  }
  //calculate error delta
  def calDelta(prevDelta:Array[Double]): Double ={
    delta = output*(1.0-output)*weight.zip(prevDelta).map(x=>x._1._2*x._2).sum
    delta
  }
  def adjustWeight(prevDelta:Array[Double]): Unit ={
    val change = prevWeight.map(x=>(x._1,momentum*x._2)).zip(prevDelta.map(x=>eta*x*output)).map(x=>(x._1._1,x._1._2+x._2))
    weight = weight.zip(change).map(x=>(x._1._1,x._1._2+x._2._2))
    prevWeight = change
  }
}
