package GPUCNN

import java.io._
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}
import scala.io.Source
import jcuda.driver.{CUfunction, CUmodule}
import jcuda.driver.JCudaDriver.cuLaunchKernel
import jcuda.driver.JCudaDriver.cuModuleLoad
import jcuda.driver.JCudaDriver.cuModuleGetFunction
import jcuda.{Pointer, Sizeof}
import jcuda.jcublas.JCublas2.{cublasCreate, cublasDestroy, cublasSgemv}
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T
import jcuda.jcudnn.JCudnn.{CUDNN_VERSION, cudnnActivationForward, cudnnAddTensor, cudnnConvolutionForward, cudnnCreate, cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor, cudnnCreateLRNDescriptor, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy, cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyLRNDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithm, cudnnGetConvolutionForwardWorkspaceSize, cudnnGetConvolutionNdForwardOutputDim, cudnnGetErrorString, cudnnGetPoolingNdForwardOutputDim, cudnnGetVersion, cudnnLRNCrossChannelForward, cudnnPoolingForward, cudnnSetConvolutionNdDescriptor, cudnnSetFilterNdDescriptor, cudnnSetLRNDescriptor, cudnnSoftmaxForward, _}
import jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID
import jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT
import jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
import jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION
import jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT
import jcuda.jcudnn.cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1
import jcuda.jcudnn.cudnnPoolingMode._
import jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE
import jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL
import jcuda.jcudnn._
import jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda._
import jcuda.jcublas.JCublas2._
import jcuda.runtime.cudaMemcpyKind.{cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice}

/**
  * A port of the "mnistCUDNN" sample.<br>
  * <br>
  * This sample expects the data files that are part of the
  * mnistCUDNN sample to be present in a "data/" subdirectory.
  */
object MnistJCudnn extends utility{
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
  var C1 = new GPUConvLayer(1,4,5,in_h,in_w,batchSize)
  var S2 = new GPUPoolLayer(2,2,4,24,24,batchSize)
  var C3 = new GPUConvLayer(4,6,5,12,12,batchSize)
  var S4 = new GPUPoolLayer(2,2,6,8,8,batchSize)
  var F5 = new GPUFullyConnectedLayer(4*4*6,150,batchSize)
  var F6 = new GPUFullyConnectedLayer(150,out_len,batchSize)

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
  def main(args:Array[String]): Unit ={
    var c = 0
    init()
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
    val dir:String = "/home/philip/Desktop/SparkNNDL-master/mnist_data/"
    trainfile = trainfile.map(x=>dir+x)

    for(i<-0 to 10){
      trainfile.foreach{f=>
        for(line<-Source.fromFile(f).getLines()){
          val splits = line.split(",").map(_.toFloat)
          val l = splits(0).toInt
          var l_arr = Array(0f,0f,0f,0f,0f,0f,0f,0f,0f,0f)
          l_arr(l) = 1f
          val d = splits.drop(1).map(_/255)
          cudaMemcpy(data,Pointer.to(d),d.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
          cudaMemcpy(label,Pointer.to(l_arr),l_arr.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
          print(c)
          networkIteration(l)
          c=c+1
        }
      }
    }

    destroy()
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

    val p = F6.fcActD
    val max_digits = 10
    val result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    //print(result.toSeq)
    if(result.max == result(l))
      println("correct")
    else
      println("wrong")
    cudaMemcpy(loss, F6.fcActD, Sizeof.FLOAT * batchSize * out_len, cudaMemcpyDeviceToDevice)
    cublasSaxpy(cbHandle,out_len*batchSize,pointerTo(-1f),label,1,loss,1)
    val scalVal = 1.0f/batchSize.toFloat
    cublasSscal(cbHandle, out_len * batchSize, pointerTo(scalVal), loss, 1)

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
  def destroy(): Unit ={
    C1.destroy()
    S2.destroy()
    C3.destroy()
    S4.destroy()
    F5.destroy()
    F6.destroy()
  }
}