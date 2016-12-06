package TestCase.CNNClassifier

import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.io.Source

/**
 * Created by philippy on 2016/3/9.
 */
class MnistLoader() {
  var sc:SparkContext = SparkContext.getOrCreate()
  var batch_size:Int = 50
  def setBatchSize(size:Int): Unit ={
    this.batch_size = size
  }
  def localLoadFile(path:String): Array[(DM[Double],DV[Double])] ={
    sc.textFile(path).collect().map(x=>format(x))
  }
  def clusterLoad(data:Array[String]): Array[RDD[(DM[Double],DV[Double])]] ={
    val collection = data.map(x=>format(x))
    repartition(collection)
  }

  def repartition(arr:Array[(DM[Double],DV[Double])]): Array[RDD[(DM[Double],DV[Double])]]={
    var res = Array[Array[(DM[Double],DV[Double])]]()
    for(i<- 0 to arr.length/batch_size-1)
      res = res++Array(arr.slice(i*batch_size,(i+1)*batch_size))
    res.map(batch=>sc.parallelize(batch))
  }

  def format(line:String): (DM[Double],DV[Double]) ={
    val parts = line.split(",")
    val label: Int = parts(0).toInt
    val data: Array[Double] = parts.drop(1).map(_.toDouble/255)
    //val labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    val labarr: Array[Double] = Array(0,0,0,0,0,0,0,0,0,0)
    labarr(label) = 1
    val input:DM[Double] = padDM(new DM(28,28,data),2)
    val target:DV[Double] = new DV(labarr)
    (input,target)
  }
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
