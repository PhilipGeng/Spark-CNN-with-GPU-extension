package TestCase.mlp

import java.io.{FileOutputStream, ObjectOutputStream}

import NNFM.multilayerPerceptron
import org.apache.spark.SparkContext
/**
 * Created by philippy on 2015/12/1.
 */
object mlptrain  {
  def main(args:Array[String]): Unit = {
    val trainfile:Array[String] = Array("data/mnist_train1","data/mnist_train2","data/mnist_train3","data/mnist_train4","data/mnist_train5","data/mnist_train6")
    val sc = new SparkContext("local", "NNtrain")
    var mlp:multilayerPerceptron = new multilayerPerceptron(28*28,Array(30,30),10)
    var cnt = 1;
    for(k<-0 to 3){
      for(i<-0 to 5) {
        println("training sample "+i)
        var trstr = sc.textFile(trainfile(i)).collect();
        trstr.foreach { line =>
          cnt=cnt+1
          val parts = line.split(",");
          val label: Int = parts(0).toInt;
          val data: Array[Double] = parts.drop(1).map(_.toDouble / 255);
          var labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          mlp.train(data, labarr);
        }
      }
    }
    val fos = new FileOutputStream("mlp30,30.out")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(mlp)
    println("your multi layer neural network is successfully saved")
    oos.close();
  }
}
