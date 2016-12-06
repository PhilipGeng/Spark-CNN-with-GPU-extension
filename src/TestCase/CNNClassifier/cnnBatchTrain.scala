package TestCase.CNNClassifier

import java.io.{FileInputStream, ObjectInputStream, FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by philippy on 2016/2/12.
 */
object cnnBatchTrain {
  var net:CNN = new CNN
  var fos: FileOutputStream = _
  var oos: ObjectOutputStream = _
  def main(args:Array[String]): Unit = {
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
    val dir:String = "hdfs:///mnist/"
    trainfile = trainfile.map(x=>dir+x)
    val conf = new SparkConf().setAppName("CNN batch train").set("spark.network.timeout","600").set("spark.akka.heartbeat.interval","100")
    val sc = new SparkContext(conf)
    var mnistLoader = new MnistLoader()
    mnistLoader.setBatchSize(20)

    // load model from file
    /* val modelPath = "/root/CNN0.out"
     val fis1 = new FileInputStream(modelPath)
     val ois1 = new ObjectInputStream(fis1)
     net = ois1.readObject.asInstanceOf[CNN]
*/
    //cluster training
    /*val cluster_ite: Int = 5
    val cluster_eta: Array[Double] = Array(0.03,0.01,0.0095,0.009,0.0085)
    val cluster_updateWhenWrong: Array[Boolean] = Array(false,false,false,true,true)
    */
    val cluster_ite: Int = 4
    val cluster_eta: Array[Double] = Array(0.01,0.0095,0.009,0.009)
    val cluster_updateWhenWrong: Array[Boolean] = Array(false,false,false,false)

    require(cluster_ite == cluster_eta.length)
    require(cluster_ite == cluster_updateWhenWrong.length)

    val start = System.nanoTime()
    fos = new FileOutputStream("CNNtrainstart.out")
    oos = new ObjectOutputStream(fos)
    oos.writeObject(net)

    //net.setNumPartition(9)

    //cluster
    for(k<-0 to cluster_ite-1){
      net.setEta(cluster_eta(k))
      net.setUpdateWhenWrong(cluster_updateWhenWrong(k))
      for(i<- 0 to trainfile.length-1){ // 10000 samples
        println("cluster training: ite "+k+" sample "+i+"0000")
        val trstr:Array[String] = sc.textFile(trainfile(i)).collect()
        mnistLoader.clusterLoad(trstr).foreach{rdd=>
          val in:RDD[DM[Double]] = rdd.map(_._1).cache()
          val t:RDD[DV[Double]] = rdd.map(_._2).cache()
          net.train(in,t)
          in.unpersist()
          t.unpersist()
        }
      }
      fos = new FileOutputStream("CNNclusterIte"+k+"t"+(System.nanoTime()-start).toString+".out")
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
