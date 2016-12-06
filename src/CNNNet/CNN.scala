package CNNNet

import CNNLayer.{CL, FL, OL, SL}
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV, sum}
import org.apache.spark.rdd.RDD


/**
  * Created by philip on 10/26/16.
  */
class CNN extends Serializable {
  val c1_fm_in_map = Array(Array(0),Array(0),Array(0),Array(0),Array(0),Array(0))
  val c3_fm_in_map = Array(Array(0,1,2),Array(1,2,3),Array(2,3,4),Array(3,4,5),Array(4,5,0),Array(5,0,1),Array(0,1,2,3),Array(1,2,3,4),Array(2,3,4,5),Array(3,4,5,0),Array(4,5,0,1),Array(5,0,1,2),Array(0,1,3,4),Array(1,2,4,5),Array(0,2,3,5),Array(0,1,2,3,4,5))
  val c5_fm_in_map = Array.fill[Array[Int]](120){(0 to 15).toArray}
  var c1:CL = new CL(6,5)
  c1.set_fm_input_map(c1_fm_in_map)
  var s2:SL = new SL(6,2)
  var c3:CL = new CL(16,5)
  c3.set_fm_input_map(c3_fm_in_map)
  var s4:SL = new SL(16,2)
  var c5:CL = new CL(120,5)
  c5.set_fm_input_map(c5_fm_in_map)
  var f6:FL = new FL(120,84)
  var o7:OL = new OL(84,10)
  var network = Array(c1,s2,c3,s4,c5,f6,o7)
  var localerr: Double = 0
  var updateWhenWrong: Boolean = false
  var numPartition: Int = 2


  /**
    * single node mode
    * @param input image
    * @param target label
    */
  def train(input:RDD[DM[Double]], target:RDD[DV[Double]]): Unit ={
    var target_filtered:RDD[DV[Double]] = target
    val fpRes:RDD[DV[Double]] = fpRDD(input)
    localerr = fpRes.zip(target).map(v=>o7.loss(v._1,v._2)).collect().sum
    if(updateWhenWrong){
      target_filtered = fpRes.zip(target).map{ite=>
        if(judgeRes(ite._1,ite._2))
          ite._1 //if classification correct, set target = classified, so that err = 0
        else
          ite._2 //if classification wrong, set target = target, err!=0
      }
    }

    o7.calErr(target_filtered)
    f6.calErr(o7)
    c5.calErr(f6)
    s4.calErr(c5)
    c3.calErr(s4)
    s2.calErr(c3)
    c1.calErr(s2)

    c1.adjWeight()
    s2.adjWeight()
    c3.adjWeight()
    s4.adjWeight()
    c5.adjWeight()
    f6.adjWeight()
    o7.adjWeight()

    clearAllCache()

  }

  def train(input:DM[Double], target:DV[Double]): Unit ={
    val fpRes:DV[Double] = classify(input)
    //calculate error
    localerr = o7.loss(fpRes,target)
    if(!(updateWhenWrong && judgeRes(fpRes,target))){
      o7.calErrLocal(target)
      f6.calErrLocal(o7)
      c5.calErrLocal(f6)
      s4.calErrLocal(c5)
      c3.calErrLocal(s4)
      s2.calErrLocal(c3)
      c1.calErrLocal(s2)

      c1.adjWeightLocal()
      s2.adjWeightLocal()
      c3.adjWeightLocal()
      s4.adjWeightLocal()
      c5.adjWeightLocal()
      f6.adjWeightLocal()
      o7.adjWeightLocal()
    }
  }

  def judgeRes(fpRes:DV[Double],target:DV[Double]): Boolean ={
    var res: Boolean = false
    for(i<-0 to target.length-1){
      if(target(i) == 1){
        if(fpRes.max == fpRes(i))
          res = true
      }
    }
    res
  }

  def fpRDD(input:RDD[DM[Double]]):RDD[DV[Double]] = {
    val in: RDD[Array[DM[Double]]] = input.map(Array(_))
    val c1res:RDD[Array[DM[Double]]] = c1.forward(in)
    val s2res:RDD[Array[DM[Double]]] = s2.forward(c1res)
    val c3res:RDD[Array[DM[Double]]] = c3.forward(s2res)
    val s4res:RDD[Array[DM[Double]]] = s4.forward(c3res)
    val c5res:RDD[Array[DM[Double]]] = c5.forward(s4res)
    val f6res:RDD[Array[DM[Double]]] = f6.forward(c5res)
    val o7res:RDD[Array[DM[Double]]] = o7.forward(f6res)
    o7res.map(x=>o7.flattenOutput(x))
  }
  def classify(input:RDD[DM[Double]]):RDD[DV[Double]] = {
    val res = fpRDD(input)
    clearAllCache()
    res
  }

  def classify(input:DM[Double]):DV[Double] = {
    val in: Array[DM[Double]] = Array(input)
    val c1res:Array[DM[Double]] = c1.forwardLocal(in)
    val s2res:Array[DM[Double]] = s2.forwardLocal(c1res)
    val c3res:Array[DM[Double]] = c3.forwardLocal(s2res)
    val s4res:Array[DM[Double]] = s4.forwardLocal(c3res)
    val c5res:Array[DM[Double]] = c5.forwardLocal(s4res)
    val f6res:Array[DM[Double]] = f6.forwardLocal(c5res)
    val o7res:Array[DM[Double]] = o7.forwardLocal(f6res)
    c1.flattenOutput(o7res)
  }

  def setEta(eta:Double): Unit ={
    c1.seteta(eta)
    s2.seteta(eta)
    c3.seteta(eta)
    s4.seteta(eta)
    c5.seteta(eta)
    f6.seteta(eta)
    o7.seteta(eta)
  }

  def setMomentum(m:Double): Unit ={
    c1.setMomentum(m)
    s2.setMomentum(m)
    c3.setMomentum(m)
    s4.setMomentum(m)
    c5.setMomentum(m)
    f6.setMomentum(m)
    o7.setMomentum(m)
  }

  def clearAllCache(): Unit ={
    c1.clearCache()
    s2.clearCache()
    c3.clearCache()
    s4.clearCache()
    c5.clearCache()
    f6.clearCache()
    o7.clearCache()
  }

  def setUpdateWhenWrong(a:Boolean): Unit ={
    updateWhenWrong = a
  }

  def setNumPartition(par:Int): Unit ={
    numPartition = par
    c1.setNumPartition(numPartition)
    s2.setNumPartition(numPartition)
    c3.setNumPartition(numPartition)
    s4.setNumPartition(numPartition)
    c5.setNumPartition(numPartition)
    f6.setNumPartition(numPartition)
    o7.setNumPartition(numPartition)
  }
  def setActivate(s:String): Unit ={
    c1.setActivate(s)
    s2.setActivate(s)
    c3.setActivate(s)
    s4.setActivate(s)
    c5.setActivate(s)
    f6.setActivate(s)
    o7.setActivate(s)
  }
}
