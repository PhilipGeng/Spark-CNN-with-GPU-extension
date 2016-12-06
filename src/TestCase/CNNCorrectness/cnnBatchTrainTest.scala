package TestCase.CNNCorrectness

import java.io.{ObjectInputStream, FileInputStream, FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import TestCase.CNNClassifier.MnistLoader
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/2/12.
 */
object cnnBatchTrainTest {
  var net1:CNN = new CNN
  var net2:CNN = new CNN
  def main(args:Array[String]): Unit = {
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
   // val dir:String = "hdfs:///mnist/"
    val dir:String = "/home/philippy/IdeaProjects/NeuralNetwork/data/"
    trainfile = trainfile.map(x=>dir+x)
    val conf = new SparkConf().setAppName("CNN batch train").setMaster("local")
    val sc = new SparkContext(conf)
    //cluster training
    val cluster_ite: Int = 5
    val cluster_eta: Array[Double] = Array(0.5,0.5,0.5,0.5,0.5)
    val loader:MnistLoader = new MnistLoader
    loader.setBatchSize(1)

    val start = System.nanoTime()
    var fos = new FileOutputStream("CNNtrain.out")
    var oos = new ObjectOutputStream(fos)
    val modelPath = "CNNcluster0.out"
    val fis1 = new FileInputStream(modelPath)
    val ois1 = new ObjectInputStream(fis1)
    net1 = ois1.readObject.asInstanceOf[CNN]
    val fis2 = new FileInputStream(modelPath)
    val ois2 = new ObjectInputStream(fis2)
    net2 = ois2.readObject.asInstanceOf[CNN]
    net1.setMomentum(0.9d)
    net2.setMomentum(0.9d)
 //   net.setUpdateWhenWrong(true)
 //   net.setNumPartition(9)
/*    //cluster
    for(k<-0 to 0){
    //for(k<-0 to cluster_ite-1){
      net.setEta(cluster_eta(k))
      for(i<- 0 to trainfile.length-1){ // 10000 samples
        println("cluster training: ite "+k+" sample "+i+"0000")
        val trstr = sc.textFile(trainfile(i)).collect()
        loader.clusterLoad(trstr).foreach{rdd=>
          val in:RDD[DM[Double]] = rdd.map(_._1).cache()
          val t:RDD[DV[Double]] = rdd.map(_._2).cache()
          net.train(in,t)
          println("cluster train finished 1 iteration")
        }
         }
      fos = new FileOutputStream("CNNclusterIte"+k+"t"+(System.nanoTime()-start).toString+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved cluster ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
      }*/
  val trstr = sc.textFile(trainfile(0)).collect()
  val s0in = loader.format(trstr(0))._1
  val s0la = loader.format(trstr(0))._2
  val s1in = loader.format(trstr(1))._1
  val s1la = loader.format(trstr(1))._2
  val s2in = loader.format(trstr(2))._1
  val s2la = loader.format(trstr(2))._2
  val cin = sc.parallelize(Array(s0in,s1in,s2in),2)
  val cla = sc.parallelize(Array(s0la,s1la,s2la),2)
    val c0in = sc.parallelize(Array(s0in),2)
    val c0la = sc.parallelize(Array(s0la),2)
    val c1in = sc.parallelize(Array(s1in),2)
    val c1la = sc.parallelize(Array(s1la),2)
    val c2in = sc.parallelize(Array(s2in),2)
    val c2la = sc.parallelize(Array(s2la),2)

    val o = net1.o7.weight(3,3)
    net1.train(s0in,s0la)
    val w1 = net1.o7.weight(3,3)
   // net1.train(s1in,s1la)
   // val w2 = net1.s2.wadj(2)
   // net1.train(s2in,s2la)
   // val w3 = net1.s2.wadj(2)
   // println("first:"+(w1+w2+w3))
   // net.train(s1in,s1la)
    //println(net.c1.deltaLocal(0))
   // net2.train(cin,cla)
   // println(net2.s2.wadj(2))
    net2.train(c0in,c0la)
    val c1 = net2.o7.weight(3,3)
    println(w1)
    println(c1)
    println(o)
    /*net2.train(c1in,c1la)
    val c2 = net2.s2.wadj(1)
    net2.train(c2in,c2la)
    val c3 = net2.s2.wadj(1)
    println("second"+(c1+c2+c3))*/
  }

  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
