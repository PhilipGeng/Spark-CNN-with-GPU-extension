package NNFM

/**
 * Created by philippy on 2015/12/14.
 */
import breeze.linalg.{*, DenseMatrix => DM, DenseVector => DV}
import breeze.numerics.sigmoid

class multilayerPerceptron(val input:Int, val hid:Array[Int], val output:Int, val eta: Double=0.5, val momentum: Double=0.1) extends Serializable{

  val inputlayer = new DV[Double](input+1)
  val hiddenlayer :Array[DV[Double]] = hid.map(x=>new DV[Double](x+1))
  val outputlayer = new DV[Double](output+1)
  var reallabel = new DV[Double](output+1)

  var layers:Array[DV[Double]] = Array(inputlayer)++hiddenlayer++Array(outputlayer)

  var delta:Array[DV[Double]] = hiddenlayer++Array(outputlayer)

  var weight: Array[DM[Double]] = new Array(layers.length-1)
  var prevweight: Array[DM[Double]] = new Array(layers.length-1)

  for(i<-0 to weight.length-1){
    weight(i) = DM.fill[Double](layers(i).length,layers(i+1).length){scala.util.Random.nextDouble()*2-1d}
    prevweight(i) = DM.fill[Double](layers(i).length,layers(i+1).length){0d}
  }

  def train(features:Array[Double], labels:Array[Double]): Unit ={
    loadData(features,labels)
    forwardProp()
    backProp()
  }

  def predict(features:Array[Double]): Array[Double] ={
    loadFeature(features)
    forwardProp()
    layers(layers.length-1).toArray.drop(1)
  }

  def loadFeature(features:Array[Double]): Unit ={
    if(layers(0).length!=features.length+1)
      println("load data error: dimension not fit")
    layers(0) = new DV(Array(1d)++features)
  }

  def loadData(features:Array[Double], labels:Array[Double]): Unit ={
    if(layers(0).length!=features.length+1)
      println("error feature")
    if(reallabel.length!=labels.length+1)
      println("error label")
    layers(0) = new DV(Array(1d)++features)
    reallabel = new DV(Array(1d)++labels)
  }

  def forwardProp(): Unit ={
    for(i<- 1 to layers.length-1){
      layers(i) = forward(layers(i-1),weight(i-1))
    }
  }

  def forward(prevlayer:DV[Double],  weight:DM[Double]): DV[Double] ={
    prevlayer(0)=1.0
    var res:DV[Double] = (prevlayer.t * weight).t
    for(x<- 0 until res.length){
      res(x) = sigmoid(res(x))
    }
    res
  }

  def backProp(): Unit ={
    outputErr()
    hiddenErr()
    adjustWeight()
  }

  def outputErr(): Unit ={
    layers(layers.length-1)(0) = 1d
    val output = layers(layers.length-1)
    delta(delta.length-1) = output :* (output :- 1d) :* (output :- reallabel)
    delta(delta.length-1)(0) = 0
  }

  def hiddenErr(): Unit ={
    for(x<- delta.length-2 to 0 by -1){
      val sum:DV[Double]= weight(x+1)*delta(x+1)
      val o:DV[Double] = layers(x+1)
      delta(x) = o :* (o:-1d) :* sum :* -1d
      delta(x)(0) = 0
    }
  }

  def adjustWeight(): Unit ={
    for(i<- weight.length-1 to 0 by -1){
      prevweight(i) = calAdj(i)
      weight(i) = weight(i):+prevweight(i)
    }
  }

  def calAdj(i:Int): DM[Double] ={
    (layers(i)*(delta(i).t):*eta):+(prevweight(i):*momentum)
  }
}
