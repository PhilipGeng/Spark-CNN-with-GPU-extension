package CNNLayer

import breeze.linalg.{sum, DenseMatrix => DM, DenseVector => DV}
import breeze.numerics._
import org.apache.spark.rdd.RDD

/**
  * Created by philip on 10/26/16.
  */
trait layer extends Serializable {
  /** default values */
  var eta: Double = 0.01
  var momentum: Double = 0.9
  var actfunc: String = "sigmoid" //enum{sigmoid,relu,tanh}

  /**interfaces for cluster mode*/
  var numPartition: Int = 2
  def setNumPartition(par:Int): Unit ={numPartition = par}

  /**
    * setter function for learning rate
    */
  def seteta(e:Double) = {
    this.eta = e
  }
  /**
    * setter function for momentum
    */
  def setMomentum(m:Double)={
    this.momentum = m
  }
  /**
    * setter function for activation
    */
  def setActivate(s:String): Unit={
    if(s=="sigmoid"||s=="relu"||s=="tanh"){
      actfunc = s
    }
    else{
      println("error setting activation function, support: sigmoid/relu/tanh")
    }
  }
  /**
    * format output for intra-class calculation
    * @param output: Array of DM
    * @return DV[Double]
    */
  def flattenOutput(output: Array[DM[Double]]): DV[Double] ={
    val flatOutput: DV[Double] = new DV(output.map(x=>x(0,0)))
    flatOutput
  }

  /**
    * format output for inter-class communication
    * @param output: DV[Double]
    * @return Array of DM
    */
  def formatOutput(output: DV[Double]): Array[DM[Double]] ={
    val format:Array[DM[Double]] = output.toArray.map(x=>DM.fill(1,1){x})
    format
  }

  /**
    * for example:
    * Mat = [1 2
    *        3 4]
    * expandDM(Mat,2) = [1 1 2 2
    *                    1 1 2 2
    *                    3 3 4 4
    *                    3 3 4 4]
    */
  def expandDM(Mat:DM[Double],scalar:Int): DM[Double] ={
    DM.tabulate(Mat.rows*scalar,Mat.cols*scalar){case(i,j)=>Mat(i/scalar,j/scalar)}
  }

  /**
    * convolve operation for matrix
    * @param Mat matrix dim=m,m
    * @param kernel convolve kernel dim=k,k
    * @return result dim=m-k+1,m-k+1
    */
  def convolveDM(Mat:DM[Double],kernel:DM[Double]):DM[Double]={
    DM.tabulate(Mat.rows-kernel.rows+1,Mat.cols-kernel.cols+1){case(i,j)=>sum(Mat(i to i+kernel.rows-1,j to j+kernel.cols-1):*kernel)}
  }

  /**
    * rotate matrix by 180 degrees
    * for example, mat = [1 2 3
    *                     4 5 6
    *                     7 8 9]
    *       rot180(mat) = [9 8 7
    *                      6 5 4
    *                      3 2 1]
    * @param in matrix
    * @return rotated matrix
    */
  def rot180(in:DM[Double]):DM[Double] = {
    DM.tabulate(in.rows,in.cols){case(i,j)=>in(in.rows-i-1,in.cols-j-1)}
  }

  /**
    * surround mat by number(padding) of zeros
    * @param mat original mat
    * @param padding padding number
    * @return
    */
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }

  /**
    * zip rdds to become array, deprecated due to large memory and time cost
    * @param s Array of RDD
    * @return RDD of Array
    */
  def makeZip(s: Array[RDD[DM[Double]]]): RDD[Array[DM[Double]]] = {
    if (s.length == 1)
      s.head.map(e => Array(e))
    else {
      val others:RDD[Array[DM[Double]]] = makeZip(s.tail)
      val all: RDD[(DM[Double],Array[DM[Double]])] = s.head.zip(others)
      all.map(elem => Array(elem._1) ++ elem._2)
    }
  }

  /**
    * unzip array to rdd, deprecated due to large memory and time cost
    * @param s RDD of array
    * @param numOfCol dimension of Array
    * @return array of RDD
    */
  def unzip(s: RDD[Array[DM[Double]]], numOfCol: Int): Array[RDD[DM[Double]]] = {
    (0 until numOfCol).toArray.map{index=>s.map{arr=>arr(index)}}
  }

  def activate(x:Double): Double ={
    actfunc match {
      case "relu"=>Math.max(0d,x)
      case "tanh"=>tanh(x)
      case "sigmoid"=>sigmoid(x)
    }
  }
  def activate(x:DM[Double]): DM[Double] ={
    x.map(a=>activate(a))
  }
  def activate(x:DV[Double]): DV[Double] ={
    x.map(a=>activate(a))
  }
  def act_derivative(o:Double): Double ={
    actfunc match {
      case "relu"=>if(o>0d) 1d else 0d
      case "tanh"=>1-o*o
      case "sigmoid"=>o*(1-o)
    }
  }
  def act_derivative(o:DM[Double]): DM[Double] ={
    actfunc match {
      case "relu"=>o.map(x=>if(x>0d) 1d else 0d)
      case "tanh"=>((o:*o):-1d):*(-1d)
      case "sigmoid"=>o:*(o:-1d):*(-1d)
    }
  }
  def act_derivative(o:DV[Double]): DV[Double] ={
    actfunc match {
      case "relu"=>o.map(x=>if(x>0d) 1d else 0d)
      case "tanh"=>((o:*o):-1d):*(-1d)
      case "sigmoid"=>o:*(o:-1d):*(-1d)
    }
  }

}
