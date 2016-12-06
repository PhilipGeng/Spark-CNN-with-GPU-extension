package GPUCNN

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import jcuda.{Pointer, Sizeof}
import jcuda.jcublas.JCublas2._
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.jcudnn.JCudnn._
import jcuda.jcudnn.cudnnActivationMode._
import jcuda.jcudnn.cudnnDataType._
import jcuda.jcudnn.{JCudnn, cudnnHandle, cudnnTensorDescriptor}
import jcuda.jcudnn.cudnnTensorFormat._
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._

import scala.io.Source

/**
  * Created by philip on 10/31/16.
  */
class GPUCNN extends utility with Serializable{
  val batchSize = 1
  JCuda.setExceptionsEnabled(true)
  JCudnn.setExceptionsEnabled(true)
  JCublas2.setExceptionsEnabled(true)
  val version: Int = cudnnGetVersion.toInt
  println("cudnnGetVersion() : "+version+" , " + "CUDNN_VERSION from cudnn.h :"+ CUDNN_VERSION)
  var cuHandle: cudnnHandle = new cudnnHandle
  var cbHandle: cublasHandle = new cublasHandle
  cudnnCreate(cuHandle)
  cublasCreate(cbHandle)

  val in_h = 28
  val in_w = 28
  val out_len = 10
  var C1 = new GPUConvLayer(1,32,5,in_h,in_w,batchSize)
  var S2 = new GPUPoolLayer(2,2,32,24,24,batchSize)
  var C3 = new GPUConvLayer(32,64,5,12,12,batchSize)
  var S4 = new GPUPoolLayer(2,2,64,8,8,batchSize)
  var F5 = new GPUFullyConnectedLayer(4*4*64,1024,batchSize)
  var F6 = new GPUFullyConnectedLayer(1024,out_len,batchSize)

  var dataT: cudnnTensorDescriptor = new cudnnTensorDescriptor
  cudnnCreateTensorDescriptor(dataT)
  cudnnSetTensor4dDescriptor(dataT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batchSize,1,in_h,in_w)
  var data: Pointer = createDevicePointer(Array.fill[Float](in_h*in_w){1f})
  var label: Pointer = createDevicePointer(Array.fill[Float](10){1f})

  var loss:Pointer = new Pointer()
  cudaMalloc(loss,batchSize*out_len*Sizeof.FLOAT)
  F5.setActivation(CUDNN_ACTIVATION_SIGMOID)
  F6.setActivation(-1)
  var zeroVec = Array(0d,0d,0d,0d,0d,0d,0d,0d,0d,0d)
  init()

  def networkTrain(data_host:Array[Float],label_host:Int): Unit ={
    var l_arr = Array(0f,0f,0f,0f,0f,0f,0f,0f,0f,0f)
    l_arr(label_host) = 1f
    cudaMemcpy(data,Pointer.to(data_host),data_host.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
    cudaMemcpy(label,Pointer.to(l_arr),l_arr.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
    networkIteration(label_host)
  }

  def networkTest(data_host:Array[Float]): Int ={
    cudaMemcpy(data,Pointer.to(data_host),data_host.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
    forward()
    val p = F6.fcActD
    val max_digits = 10
    val result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    val r = result.indexOf(result.max)
    r
  }

  def networkIteration(l:Int): Unit ={
    forward()
    softmaxLoss(l)
    backward()
    update()
  }

  def init(): Unit ={
    C1.init(cuHandle,cbHandle,data,dataT,S2.srcDiff)
    S2.init(cuHandle,C1.convD,C1.convT,C3.srcDiff)
    C3.init(cuHandle,cbHandle,S2.poolD,S2.poolT,S4.srcDiff)
    S4.init(cuHandle,C3.convD,C3.convT,F5.srcDiff)
    F5.init(cuHandle,cbHandle,S4.poolD,S4.poolT,F6.srcDiff)
    F6.init(cuHandle,cbHandle,F5.fcActD,F5.fcT,loss)
  }
  def forward(): Unit ={
    C1.forwardPropagation()
    S2.forwardPropagation()
    C3.forwardPropagation()
    S4.forwardPropagation()
    F5.forwardPropagation()
    F6.forwardPropagation()
  }
  def softmaxLoss(l:Int): Unit ={
    cudaMemcpy(loss, F6.fcActD, Sizeof.FLOAT * batchSize * out_len, cudaMemcpyDeviceToDevice)
    cublasSaxpy(cbHandle,out_len*batchSize,pointerTo(-1f),label,1,loss,1)

    val scalVal = 1.0f/batchSize.toFloat
    cublasSscal(cbHandle, out_len * batchSize, pointerTo(scalVal), loss, 1)
    var p = loss
    var max_digits = 10
    var result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    //print(result.map(x=>x*x).sum)
    p = F6.fcActD
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    //if(result.max == result(l))
    //  println("correct")
    //else
    //   println("wrong")
  }
  def backward(): Unit ={
    F6.backwardPropagation()
    F5.backwardPropagation()
    S4.backwardPropagation()
    C3.backwardPropagation()
    S2.backwardPropagation()
    C1.backwardPropagation()
  }
  def update(): Unit ={
    C1.updateWeight()
    C3.updateWeight()
    F5.updateWeight()
    F6.updateWeight()
  }

  def save(): Unit ={
    var fos: FileOutputStream = new FileOutputStream("CNNGPU.out")
    var oos: ObjectOutputStream = new ObjectOutputStream(fos)
    oos.writeObject(this)
  }

  def destroy(): Unit ={
    C1.destroy()
    S2.destroy()
    C3.destroy()
    S4.destroy()
    F5.destroy()
    F6.destroy()
  }
}
