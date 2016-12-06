package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is the generic layer model contains functions in common
 */
import breeze.linalg.{DenseMatrix=>DM,DenseVector=>DV,sum}
import breeze.numerics.{sigmoid,tanh}
import org.apache.spark.rdd.RDD

trait layerFunc  extends Serializable {


  def forward(input_arr:RDD[Array[DM[Double]]]):RDD[Array[DM[Double]]]
  def calErr(nextlayer:CL) : Unit
  def calErr(nextlayer:SL) : Unit
  def calErr(nextlayer:FL) : Unit
  def calErr(target: RDD[DV[Double]]): Unit
  def adjWeight(): Unit
  def clearCache(): Unit


  /**interfaces for local mode*/
  def forwardLocal(input_arr:Array[DM[Double]]):Array[DM[Double]]
  def calErrLocal(nextLayer:CL): Unit
  def calErrLocal(nextLayer:SL): Unit
  def calErrLocal(nextLayer:FL): Unit
  def calErrLocal(target: DV[Double]): Unit
  def adjWeightLocal(): Unit


}
