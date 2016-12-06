package NNFM


class mlnn (val input:Int, val hid:Array[Int], val output:Int, val eta: Double=0.25, val momentum: Double=0.9) extends Serializable{

  var inputlayer = new Array[Double](input+1);
  var hiddenlayer :Array[Array[Double]] = hid.map(x=>new Array[Double](x+1))
  var outputlayer = new Array[Double](output+1);
  var reallabel = new Array[Double](output+1);

  var layers:Array[Array[Double]] = Array(inputlayer)++hiddenlayer++Array(outputlayer)

  var delta = hiddenlayer++Array(outputlayer);

  var weight: Array[Array[Array[Double]]] = new Array(layers.length-1);
  var prevweight: Array[Array[Array[Double]]] = new Array(layers.length-1);

  for(i<- 0 to layers.length-2){
    weight(i) = Array.fill(layers(i).length,layers(i+1).length){scala.util.Random.nextDouble()*2-1d}
    prevweight(i) = Array.fill(layers(i).length,layers(i+1).length){0d}
  }

  var outErr:Double = 0d;
  var hidErr:Double = 0d;


  def train(features:Array[Double], labels:Array[Double]): Unit ={
    loadData(features,labels);
    forwardProp();
    backProp();
  }

  def predict(features:Array[Double]): Array[Double] ={
    loadData(features);
    forwardProp();
    layers(layers.length-1).drop(1);
  }

  def loadData(features:Array[Double]): Unit ={
    if(layers(0).length!=features.length+1)
      println("load data error: dimension not fit")
    layers(0) = Array(1d)++features;
  }

  def loadData(features:Array[Double], labels:Array[Double]): Unit ={
    if(layers(0).length!=features.length+1)
      println("error feature")
    if(reallabel.length!=labels.length+1)
      println("error label")
    layers(0) = Array(1d)++features;
    reallabel = Array(0d)++labels;
  }

  def forwardProp(): Unit ={
    for(i<- 1 to layers.length-1){
      layers(i) = forward(layers(i-1),weight(i-1))
    }
  }

  def forward(prevlayer:Array[Double],  weight:Array[Array[Double]]): Array[Double] ={
    prevlayer(0)=1.0;
    transpose(weight.zip(prevlayer).map(x=>x._1.map(e=>e*x._2))).map(x=>sigmoid(x.sum));
  }

  def backProp(): Unit ={
    outputErr();
    hiddenErr();
    adjustWeight();
  }

  def outputErr(): Unit ={
    reallabel(0) = layers(layers.length-1)(0);
    delta(delta.length-1) = layers(layers.length-1).zip(reallabel).map(x=>x._1*(1d-x._1)*(x._2-x._1));
    delta(delta.length-1)(0) = 0;
    outErr = delta(delta.length-1).sum;
  }

  def hiddenErr(): Unit ={
    hidErr = 0;
    for(x<- delta.length-2 to 0 by -1){
      for(i<- 1 to delta(x).length-1){
        var sum = weight(x+1)(i).zip(delta(x+1)).map(x=>x._1*x._2).sum
        var o = layers(x+1)(i);
        delta(x)(i) = o*(1d-o)*sum;
        hidErr = hidErr+Math.abs(delta(x)(i))
      }
    }
  }

  def adjustWeight(): Unit ={
    for(i<- layers.length-1 to 1 by -1){
      reweight(delta(i-1),layers(i-1),weight(i-1),prevweight(i-1));
    }
  }

  def reweight(delta:Array[Double], layer:Array[Double], weight:Array[Array[Double]], prevweight:Array[Array[Double]]): Unit ={
    layer(0)=1;
    for(i<-1 to delta.length-1){
      for(j<-0 to layer.length-1){
        var adj = momentum*prevweight(j)(i)+eta*delta(i)*layer(j);
        weight(j)(i) = weight(j)(i)+adj;
        prevweight(j)(i) = adj;
      }
    }
  }

  def transpose(xss: Array[Array[Double]]): Array[Array[Double]] = {
    for (i <- Array.range(0, xss(0).length)) yield
    for (xs <- xss) yield xs(i)
  }


  def sigmoid(value: Double):Double = {
    1d / (1d + Math.exp(-value));
  }
}
