package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This fully connected layer in CNN (multi-layer perceptron)
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import org.apache.spark.rdd.RDD
import breeze.numerics.{atan, tanh, sigmoid}

/**
 * constructor
 * @param num_in number of incoming features
 * @param num_out number of output vector
 */
class FL (val num_in:Int, val num_out:Int) extends layer with layerFunc{
  var weight:DM[Double] = DM.fill[Double](num_in, num_out){scala.util.Random.nextDouble()*2-1d}
  var bias: DV[Double] = DV.fill[Double](num_out){scala.util.Random.nextDouble()*2-1d}
  var delta: RDD[DV[Double]] = _
  var output: RDD[DV[Double]] = _
  var input: RDD[DV[Double]] = _
  var inputLocal: DV[Double] = _
  var outputLocal: DV[Double] = _
  var deltaLocal: DV[Double] = _
  var wadj:DM[Double] = DM.fill[Double](num_in,num_out){0d}
  var badj:DV[Double] = DV.fill[Double](num_out){0d}

  /**
   * override forward function
   * @param in_vec input vector(learned features) in matrix format (to be transformed)
   * @return output
   */
  def forward(in_vec:RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] ={
    //transform Array[DM[Double]] => DV[Double]
    input = in_vec.map(flattenOutput(_)).cache()
    //dot product to get output
    output = input.map { vec=> activate((vec.t * weight).t + bias)}
    //format output DV[Double] => Array[DM[Double]]
    output.cache()
    output.map(formatOutput(_))
  }

  def forwardLocal(in_vec:Array[DM[Double]]): Array[DM[Double]] = {
      //transform Array[DM[Double]] => DV[Double]
      inputLocal = flattenOutput(in_vec)
      //dot product to get output
      outputLocal = activate((inputLocal.t * weight).t + bias)
      //format output DV[Double] => Array[DM[Double]]
      formatOutput(outputLocal)
  }
  /**
   * calculate Err for this FL if it's before a FL
   * @param nextDelta
   * @param nextWeight
   */
  def calErrBeforeFl(nextDelta:RDD[DV[Double]], nextWeight:DM[Double]): Unit ={
    delta = nextDelta.zip(output).map{vec=>
      val deltaVec:DV[Double] = vec._1
      val outputVec:DV[Double] = vec._2
      val sum:DV[Double] = nextWeight*deltaVec
      act_derivative(outputVec):*sum
    }.cache()
  }
  def calErrBeforeFlLocal(nextDelta:DV[Double], nextWeight:DM[Double]): Unit ={
    val sum:DV[Double]= nextWeight*nextDelta
    val o:DV[Double] = outputLocal
    require(o.length==sum.length)
    deltaLocal = act_derivative(o) :* sum
  }

  /**
   * task distributor
   * @param nextLayer
   */
  def calErr(nextLayer:FL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)
  def calErr(nextLayer:OL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)
  def calErrLocal(nextLayer:FL): Unit = calErrBeforeFlLocal(nextLayer.deltaLocal,nextLayer.weight)
  def calErrLocal(nextLayer:OL): Unit = calErrBeforeFlLocal(nextLayer.deltaLocal,nextLayer.weight)

  /**
   * adjust weight
   * single node mode
   */
  def adjWeight(): Unit ={
    val adj:RDD[(DM[Double],DV[Double])] = delta.zip(input).map{vec=>
      val d:DV[Double] = vec._1
      val i:DV[Double] = vec._2
      val adjw:DM[Double] = i*d.t:*eta
      val adjb:DV[Double] = d:*eta
      (adjw,adjb)
    }

    val redadj:(DM[Double],DV[Double]) = adj.reduce{(i,j)=>(i._1:+j._1,i._2+j._2)}
    wadj = momentum*wadj+redadj._1
    badj = momentum*badj+redadj._2
    weight = weight:+wadj
    bias = bias:+badj
  }

  def adjWeightLocal(): Unit ={
    val adjw:DM[Double] = inputLocal*deltaLocal.t:*eta
    wadj = momentum*wadj+adjw
    weight = weight:+wadj
    val adjb:DV[Double] = deltaLocal:*eta
    badj = momentum*badj+adjb
    bias = bias:+badj
  }

  def clearCache(): Unit ={
    input.unpersist()
    output.unpersist()
    delta.unpersist()
  }

  def calErr(t:RDD[DV[Double]]):Unit={}
  def calErr(t:SL):Unit={}
  def calErr(t:CL):Unit={}

  def calErrLocal(nextLayer: CL): Unit = {}
  def calErrLocal(nextLayer: SL): Unit = {}
  def calErrLocal(target: DV[Double]): Unit = {}
}
