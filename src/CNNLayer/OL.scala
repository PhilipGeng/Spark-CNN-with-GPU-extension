package CNNLayer
/**
 * @author Philip GENG
 * gengxu.heartfire@gmail.com
 * This is fully connected layer as the output layer in CNN (multi-layer perceptron)
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM}
import breeze.numerics.sigmoid
import org.apache.spark.rdd.RDD
/**
 * constructor
 * @param num_in number of incoming features
 * @param num_out number of output vector
 */
class OL (override val num_in:Int, override val num_out:Int) extends FL(num_in, num_out){
  /**
   * override calculation of Delta, cuz it doesn't have next layer
   * @param target desired output
   */
  override def calErr(target: RDD[DV[Double]]): Unit ={
    delta = target.zip(output).map{vec=>
      val t:DV[Double] = vec._1
      val o:DV[Double] = vec._2
      act_derivative(o):*(t-o)
    }.cache()
  }
  override def calErrLocal(target: DV[Double]): Unit ={
    deltaLocal = act_derivative(outputLocal) :* (target-outputLocal)
  }
  def loss(output:DV[Double],target:DV[Double]): Double ={
    (target-output).map(x=>x*x).sum*0.5
  }

}