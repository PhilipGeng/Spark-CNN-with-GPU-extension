package GPUCNN

import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

import jcuda.driver.{CUfunction, CUmodule}
import jcuda.{Pointer, Sizeof}
import jcuda.jcublas.JCublas2.{cublasCreate, cublasDestroy, cublasSgemv}
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T
import jcuda.jcudnn.JCudnn.{CUDNN_VERSION, cudnnActivationForward, cudnnAddTensor, cudnnConvolutionForward, cudnnCreate, cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor, cudnnCreateLRNDescriptor, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy, cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyLRNDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithm, cudnnGetConvolutionForwardWorkspaceSize, cudnnGetConvolutionNdForwardOutputDim, cudnnGetErrorString, cudnnGetPoolingNdForwardOutputDim, cudnnGetVersion, cudnnLRNCrossChannelForward, cudnnPoolingForward, cudnnSetConvolutionNdDescriptor, cudnnSetFilterNdDescriptor, cudnnSetLRNDescriptor, cudnnSoftmaxForward, _}
import jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU
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
object test extends utility{
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

  /*
    val in_h = 32
    val in_w = 32
    val out_len = 10

    var C1 = new GPUConvLayer(1,20,5,in_h,in_w,batchSize)
    var S2 = new GPUPoolLayer(2,2,20,28,28,batchSize)
    var C3 = new GPUConvLayer(20,50,5,14,14,batchSize)
    var S4 = new GPUPoolLayer(2,2,50,10,10,batchSize)
    var F5 = new GPUFullyConnectedLayer(S4.out_width*S4.out_height*C3.out_channels/4,500,batchSize)
    var F6 = new GPUFullyConnectedLayer(F5.outputs,out_len,batchSize)
  */

  var dataT: cudnnTensorDescriptor = new cudnnTensorDescriptor
  cudnnCreateTensorDescriptor(dataT)
  cudnnSetTensor4dDescriptor(dataT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batchSize,1,6,6)

  //var data_arr = (1 to 36).toArray.map(_.toFloat)
  var data_arr = Array.fill[Float](36){1f}
  var label_arr = Array(1,0,0,0,0,0,0,0,0,0).map(_.toFloat)
  var data: Pointer = createDevicePointer(data_arr)
  var label: Pointer = createDevicePointer(label_arr)

  //var label:Pointer = createDevicePointer(l_arr)
  var loss:Pointer = new Pointer()
  cudaMalloc(loss,batchSize*3*3*Sizeof.FLOAT)
  //F5.setActivation(CUDNN_ACTIVATION_RELU)
  //F6.setActivation(-1)

  def main(args:Array[String]): Unit ={
    var c = new GPUConvLayer(1,3,3,6,6,1)
    var c2 = new GPUConvLayer(3,5,3,4,4,1)
    c.init(cuHandle,cbHandle,data,dataT,loss)
    c2.init(cuHandle,cbHandle,c.convD,c.convT,loss)
    c.forwardPropagation()
    c2.forwardPropagation()
    var p = c.convD
    var max_digits = 3*4*4
    var result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    println(result.toSeq)

    p = c2.convD
    max_digits = 5*4
    result = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    println(result.toSeq)

  }
}