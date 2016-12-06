package GPUCNN

import scala.io.Source

/**
  * Created by philip on 10/26/16.
  */
object LenetGPUBatch extends utility {
  var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
  val dir:String = "/home/philip/Desktop/SparkNNDL-master/mnist_data/"
  var testFile = "mnist_test.csv"
  var batchSize = 40
  var t1 = java.lang.System.currentTimeMillis()

  var lenet = new GPUCNNBatch(batchSize)
  def main(args:Array[String]): Unit ={
    var c = 0
    trainfile = trainfile.map(x=>dir+x)
    testFile = dir + testFile
    var max_epoch = 5
    //training phase
    for(i<-0 to max_epoch){
      print(i+": ")
      trainfile.foreach{f=>
        var image = Array[Float]()
        var label = Array[Int]()
        for(line<-Source.fromFile(f).getLines()){
          val splits = line.split(",").map(_.toFloat)
          val l = splits(0).toInt
          val d = splits.drop(1).map(_/255)
          image ++=d
          label ++=Array(l)
          //lenet.networkTrain(d,l)
          c+=1
          if(c%batchSize == 0){
            lenet.networkTrain(image,label)
            image = Array[Float]()
            label = Array[Int]()
          }
        }
      }
      println(test())
    }
    var t2 = java.lang.System.currentTimeMillis()
    //testing phase
    println("time consumption: "+(t2-t1))
    lenet.destroy()
  }

  def test(): Float ={
    var total = 0
    var correct = 0
    for(line<-Source.fromFile(testFile).getLines()){
      val splits = line.split(",").map(_.toFloat)
      val l = splits(0).toInt
      val d = splits.drop(1).map(_/255)
      val predicted = lenet.networkTest(d)
      total = total+1
      if(predicted == l)
        correct = correct+1
    }
    correct.toFloat/total.toFloat
  }
}
