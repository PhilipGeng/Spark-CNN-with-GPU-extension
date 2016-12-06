package TestCase.CNNClassifier

import java.io.{OutputStreamWriter, BufferedWriter, FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/2/12.
 */
object cnntrain {
  var net:CNN = new CNN
  def main(args:Array[String]): Unit = {
    val trainfile:Array[String] = Array("file:/root/data/mnist_train1","file:/root/data/mnist_train2","file:/root/data/mnist_train3","file:/root/data/mnist_train4","file:/root/data/mnist_train5","file:/root/data/mnist_train6")
    //val sc = new SparkContext("local", "NNtrain")
    val conf = new SparkConf().setAppName("CNN batch train")
    val sc = new SparkContext(conf)
    //local training
    val local_ite: Int = 3
    val local_eta: Array[Double] = Array(0.5,0.5,0.5)
    //cluster training
    val cluster_ite: Int = 5
    val cluster_eta: Array[Double] = Array(0.3,0.3,0.3,0.3,0.3)

    val start = System.nanoTime()
    var fos = new FileOutputStream("CNNtrain.out")
    var oos = new ObjectOutputStream(fos)

    //local training
    for(k<-0 to local_ite-1){ // 1 iterations = 60000 samples
      net.setEta(local_eta(k))
      for(i<- 0 to trainfile.length-1){ // 10000 samples
        println("local training: ite "+k+" sample "+i+"0000")
        val trstr: Array[String] = sc.textFile(trainfile(i)).collect()
        trstr.foreach{line=>
          val parts = line.split(",")
          val label: Int = parts(0).toInt
          val data: Array[Double] = parts.drop(1).map(_.toDouble/255)
          val labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input:DM[Double] = padDM(new DM(28,28,data),2)
          val target:DV[Double] = new DV(labarr)
          net.train(input,target)
        }
      }
      fos = new FileOutputStream("CNNlocal"+k+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved local ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
    }

    //cluster
    for(k<-0 to cluster_ite-1){
      net.setEta(cluster_eta(k))
      for(i<- 0 to trainfile.length-1){
        println("cluster training: ite "+k+" sample "+i+"0000")
        val trstr: RDD[(DM[Double],DV[Double])] = sc.textFile(trainfile(i)).map{rddline=>
          val arr: Array[String] = rddline.split(",")
          val label: Int = arr(0).toInt
          val data = arr.drop(1).map(_.toDouble/255)
          val labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input:DM[Double] = padDM(new DM(28,28,data),2)
          val target:DV[Double] = new DV(labarr)
          (input,target)
        }
        val in:RDD[DM[Double]] = trstr.map(_._1)
        val t:RDD[DV[Double]] = trstr.map(_._2)
        net.train(in,t)
      }
      fos = new FileOutputStream("CNNcluster"+k+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved cluster ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
    }
    oos.close()
  }
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
