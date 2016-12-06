package TestCase.CNNClassifier

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{sum, DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by philippy on 2016/2/12.
 */
object cnnLocalTrain {
  var net:CNN = new CNN
  var fos:FileOutputStream = _
  var oos:ObjectOutputStream = _
  def main(args:Array[String]): Unit = {
    //var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
    var trainfile: Array[String] = Array("mnist_train.csv")
    val dir: String = "/home/philip/Desktop/SparkNNDL-master/mnist_data/"
    trainfile = trainfile.map(x => dir + x)
    val conf = new SparkConf().setExecutorEnv("spark.executor.memory", "2g").set("spark.driver.memory", "2g")
    val sc = new SparkContext("local", "cnnLocalTrain")
    var loader: MnistLoader = new MnistLoader()
    net.setActivate("sigmoid")
    //local training
    val local_ite: Int = 6
    val local_eta: Array[Double] = Array(0.01, 0.0095, 0.009, 0.0085, 0.008, 0.008)
    require(local_ite == local_eta.length)
    val start = System.nanoTime()
    var cnt = 0
    var correct = 0
    // load model from file
    /* val modelPath = "CNNcluster1.out"
    val fis1 = new FileInputStream(modelPath)
    val ois1 = new ObjectInputStream(fis1)
    val net:CNN = ois1.readObject.asInstanceOf[CNN]
*/
    net.setUpdateWhenWrong(false)
    //local training
    fos = new FileOutputStream("CNNtrainstart.out")
    oos = new ObjectOutputStream(fos)
    oos.writeObject(net)
    for (k <- 0 to local_ite - 1) {
      // 1 iterations = 60000 sample
      net.setEta(local_eta(k))
      for (i <- 0 to trainfile.length - 1) {
        // 10000 samples
        val trstr: Array[(DM[Double], DV[Double])] = loader.localLoadFile(trainfile(i))
       /* for (n <- 0 to 10000) {
          net.train(trstr(0)._1, trstr(0)._2)
          println("sample:" + cnt + "  err:" + net.localerr + " res: " + net.judgeRes(net.o7.outputLocal, trstr(0)._2))
        }*/

        trstr.foreach{sample=>
          cnt=cnt+1
          net.train(sample._1,sample._2)
          println("sample:"+cnt+"  err:"+net.localerr+" res: "+net.judgeRes(net.o7.outputLocal,sample._2))
        }
      }
      fos = new FileOutputStream("CNNlocal"+k+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved local ite" + k + "! time:" + (System.nanoTime() - start).toString + " --finished--")
    }
    oos.close()
  }

  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
