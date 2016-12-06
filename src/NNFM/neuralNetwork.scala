package NNFM


class neuralNetwork (val input:Int, val hid:Int, val output:Int, val eta: Double=0.25, val momentum: Double=0.9){

  val r = scala.util.Random

  var inputlayer = new Array[Double](input+1);
  var hiddenlayer = new Array[Double](hid+1)
  var outputlayer = new Array[Double](output+1);
  var reallabel = new Array[Double](output+1);

  var hiddendelta = new Array[Double](hid+1);
  var outputdelta = new Array[Double](output+1);

  var weight_inphid: Array[Array[Double]] = Array.fill(input+1,hid+1){r.nextDouble()*2-1}
  var weight_hidout: Array[Array[Double]] = Array.fill(hid+1,output+1){r.nextDouble()*2-1}

  var prevweight_inphid :Array[Array[Double]]= Array.fill(input+1,hid+1){0d}
  var prevweight_hidout :Array[Array[Double]]= Array.fill(hid+1,output+1){0d}

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
    outputlayer;
  }

  def loadData(features:Array[Double]): Unit ={
    inputlayer = features;
  }

  def loadData(features:Array[Double], labels:Array[Double]): Unit ={
    inputlayer = features;
    reallabel = labels;
  }

  def forwardProp(): Unit ={
    hiddenlayer = forward(inputlayer,weight_inphid)
    outputlayer = forward(hiddenlayer,weight_hidout)
  }

  def forward(prevlayer:Array[Double],  weight:Array[Array[Double]]): Array[Double] ={
    prevlayer(0)=1.0;
    transpose(weight.zip(prevlayer).map(x=>x._1.map(e=>e*x._2))).map(x=>sigmoid(x.sum));
  }

  def backProp(): Unit ={
    outputErr();
    hiddenErr();
    reweight(outputdelta,hiddenlayer,weight_hidout,prevweight_hidout);
    reweight(hiddendelta,inputlayer,weight_inphid,prevweight_inphid);
  }

  def outputErr(): Unit ={
    reallabel(0) = outputlayer(0);
    outputdelta = outputlayer.zip(reallabel).map(x=>x._1*(1d-x._1)*(x._2-x._1));
    outputdelta(0) = 0;
    outErr = outputdelta.sum;
  }

  def hiddenErr(): Unit ={
    hidErr = 0;
    for(i<- 1 to hiddendelta.length-1){
      var sum = weight_hidout(i).zip(outputdelta).map(x=>x._1*x._2).sum
      var o = hiddenlayer(i);
      hiddendelta(i) = o*(1d-o)*sum;
      hidErr = hidErr+Math.abs(hiddendelta(i))
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
