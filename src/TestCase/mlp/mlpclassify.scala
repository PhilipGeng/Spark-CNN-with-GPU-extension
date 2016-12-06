package TestCase.mlp

import java.io._

import NNFM.multilayerPerceptron
import org.apache.spark.SparkContext

/**
 * Created by philippy on 2015/12/2.
 */
object mlpclassify {
  def main(args:Array[String]): Unit ={

    val fis = new FileInputStream("mlp30,30.out")
    val ois = new ObjectInputStream(fis)

    val mlp:multilayerPerceptron = ois.readObject.asInstanceOf[multilayerPerceptron]

    val sc = new SparkContext("local", "NNtest")
    val testfile = "data/mnist_test.csv";
    var cnt = 0;
    var correct = 0;
    var teststr = sc.textFile(testfile).collect();
    /*    var line = teststr(255)
        val parts = line.split(",");
        val label: Int = parts(0).toInt;
        val data: Array[Double] = parts.drop(1).map(_.toDouble / 255);
        val res:Array[Double] = mlp.predict(data);
        println("label : "+label)
        if(judge(res,label)==1)
          println("correct")
        else
          println("wrong")
        res.foreach(println)
    */
    teststr.foreach { line =>
      val parts = line.split(",");
      val label: Int = parts(0).toInt;
      val data: Array[Double] = parts.drop(1).map(_.toDouble / 255);
      val res = mlp.predict(data);
      var resstr = "wrong";
      if(judge(res,label)==1){
        resstr = "correct"
        correct+=1
      }
      cnt=cnt+1
      println("testing sample "+cnt+"  "+resstr)
    }
    println("correct rate: "+correct.toDouble/cnt.toDouble)
  }
  def judge(res:Array[Double],label:Int): Int ={
    var b = 0
    if(res.max == res(label))
      b = 1
    b
  }
}
