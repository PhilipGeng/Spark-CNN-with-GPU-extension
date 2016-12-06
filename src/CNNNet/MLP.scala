package CNNNet

import CNNLayer.{FL, OL}
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV, sum}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/2/18.
 */
class MLP extends Serializable {
  var f6:FL = new FL(120,84)
  var o7:OL = new OL(84,10)

  /**
   * single node mode
   * @param input image
   * @param target label
   */
  def train(input:RDD[Array[DM[Double]]], target:RDD[DV[Double]]): Unit ={
    val fpres:RDD[DV[Double]] = classify(input)
    val a:Array[DV[Double]] = fpres.collect()
    val b:Array[DV[Double]] = target.collect()
    val err:DV[Double] = a(0)-b(0)
    val Error:Double = sum(err:*err)

    println(Error)

    o7.calErr(target)
    f6.calErr(o7)
    f6.adjWeight()
    o7.adjWeight()

  }
  def classify(input:RDD[Array[DM[Double]]]):RDD[DV[Double]] = {
    val f6res:RDD[Array[DM[Double]]] = f6.forward(input)
    val o7res:RDD[Array[DM[Double]]] = o7.forward(f6res)
    o7res.map(x=>f6.flattenOutput(x))
  }

}
