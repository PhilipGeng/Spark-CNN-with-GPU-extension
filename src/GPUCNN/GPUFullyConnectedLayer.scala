package GPUCNN

import jcuda.{Pointer, Sizeof}
import jcuda.jcudnn.JCudnn._
import jcuda.jcudnn.cudnnActivationMode._
import jcuda.jcudnn.cudnnDataType._
import jcuda.jcudnn.cudnnNanPropagation._
import jcuda.jcudnn.cudnnTensorFormat._
import jcuda.jcudnn._
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation._
import jcuda.jcublas.JCublas2.cublasSgemm
import jcuda.jcublas.JCublas2.cublasSgemv
import jcuda.jcudnn.cudnnSoftmaxAlgorithm._
import jcuda.jcudnn.cudnnSoftmaxMode._
import jcuda.jcublas.JCublas2._

/**
  * Created by philip on 10/29/16.
  */
class GPUFullyConnectedLayer(var in:Int, var out:Int, val bs: Int) extends utility with Serializable{
  val inputs:Int = in
  val outputs:Int = out
  var batchSize:Int = bs
  val outputLayer = -1
  val weights: Array[Float] = Array.fill[Float](inputs*outputs){(scala.util.Random.nextFloat()*2-1)/10}
  val bias: Array[Float] = Array.fill[Float](outputs){(scala.util.Random.nextFloat()*2-1)/10}
  var weight_d: Pointer = createDevicePointer(weights)                //weight data pointer
  var bias_d: Pointer = createDevicePointer(bias)                     //bias data pointer
  var weight_g:Pointer = createDevicePointer(weights)
  var bias_g:Pointer = createDevicePointer(bias)
  var onevec:Array[Float] = Array.fill[Float](batchSize){1f}
  var onevec_d:Pointer = createDevicePointer(onevec)
  var lr = 0.01


  cudaMalloc(weight_d,weights.length*Sizeof.FLOAT)
  cudaMalloc(bias_d,bias.length*Sizeof.FLOAT)
  cudaMalloc(weight_g,weights.length*Sizeof.FLOAT)
  cudaMalloc(bias_g,bias.length*Sizeof.FLOAT)
  cudaMalloc(onevec_d,batchSize*Sizeof.FLOAT)

  cudaMemcpy(weight_d,Pointer.to(weights),weights.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
  cudaMemcpy(bias_d,Pointer.to(bias),bias.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
  cudaMemcpy(onevec_d,Pointer.to(onevec),batchSize*Sizeof.FLOAT,cudaMemcpyHostToDevice)

  var cuHandle: cudnnHandle = _
  var cbHandle: cublasHandle = _
  // input
  var srcT: cudnnTensorDescriptor = _   //input tensor
  var srcD: Pointer = _                 //input data pointer
  var srcDim: Array[Float] = Array.fill[Float](batchSize*inputs){0f}
  var srcDiff: Pointer = createDevicePointer(srcDim)
  cudaMalloc(srcDiff,srcDim.length*Sizeof.FLOAT)
  var actDiff: Pointer = createDevicePointer(srcDim)
  cudaMalloc(actDiff,srcDim.length*Sizeof.FLOAT)

  //output
  var fcT: cudnnTensorDescriptor = new cudnnTensorDescriptor
  cudnnCreateTensorDescriptor(fcT)
  cudnnSetTensor4dDescriptor(fcT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batchSize,outputs,1,1)
  var out_vec:Array[Float] = Array.fill[Float](outputs){0.1f}
  var fcD: Pointer = createDevicePointer(out_vec)
  cudaMalloc(fcD,batchSize*out_vec.length*Sizeof.FLOAT)

  var fcActD: Pointer = createDevicePointer(out_vec)
  cudaMalloc(fcActD,batchSize*out_vec.length*Sizeof.FLOAT)

  //Layer input for delta
  var nextLayerD_d: Pointer = _

  var activation = CUDNN_ACTIVATION_RELU
  var actDesc:cudnnActivationDescriptor = null

  def setActivation(a:Int): Unit ={
    activation = a
    if(activation != outputLayer){
      actDesc = new cudnnActivationDescriptor
      cudnnCreateActivationDescriptor(actDesc)
      cudnnSetActivationDescriptor(actDesc,activation,CUDNN_PROPAGATE_NAN,0.0)
    }
  }

  def init(ch:cudnnHandle, cbh:cublasHandle, srcData:Pointer, srcTensor:cudnnTensorDescriptor, nlD:Pointer): Unit ={
    this.cuHandle = ch
    this.cbHandle = cbh
    this.srcD = srcData
    this.srcT = srcTensor
    this nextLayerD_d = nlD
  }

  def forwardPropagation(): Unit = {
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
/*
    val p = srcD
    val max_digits = 10
    val result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    println(result.toSeq)
*/
    cublasSgemm(cbHandle,CUBLAS_OP_T,CUBLAS_OP_N,outputs,batchSize,inputs,alpha,weight_d,inputs,srcD,inputs,beta,fcD,outputs)
    cublasSgemm(cbHandle,CUBLAS_OP_N,CUBLAS_OP_N,outputs,batchSize,1,alpha,bias_d,outputs,onevec_d,1,alpha,fcD,outputs)
    /*val p = fcD
    val max_digits = 10
    val result: Array[Float] = new Array[Float](max_digits)
    cudaMemcpy(Pointer.to(result), p, max_digits * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    println(result.toSeq)*/
    if(activation == outputLayer){
      cudnnSoftmaxForward(cuHandle,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,alpha,fcT,fcD,beta,fcT,fcActD)
    }
    else{
      cudnnActivationForward(cuHandle,activation,alpha,fcT,fcD,beta,fcT,fcActD)
    }
  }

  def backwardPropagation(): Unit ={
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
    if(activation == outputLayer){//softmax output
      //softmaxlossbackprop
      cublasSgemm(cbHandle,CUBLAS_OP_N,CUBLAS_OP_T,inputs,outputs,batchSize,alpha,srcD,inputs,nextLayerD_d,outputs,beta,weight_g,inputs)
      cublasSgemv(cbHandle, CUBLAS_OP_N, outputs, batchSize, alpha, nextLayerD_d, outputs, onevec_d, 1, beta, bias_g, 1)
      cublasSgemm(cbHandle, CUBLAS_OP_N, CUBLAS_OP_N, inputs, batchSize, outputs, alpha, weight_d, inputs, nextLayerD_d, outputs, beta, srcDiff, inputs)
    }
    else{//not output
      cudnnActivationBackward(cuHandle,activation,alpha,fcT,fcActD,fcT,nextLayerD_d,fcT,fcD,beta,fcT,actDiff)
      cublasSgemm(cbHandle,CUBLAS_OP_N,CUBLAS_OP_T,inputs,outputs,batchSize,alpha,srcD,inputs,actDiff,outputs,beta,weight_g,inputs)
      cublasSgemv(cbHandle,CUBLAS_OP_N,outputs,batchSize,alpha,actDiff,outputs,onevec_d,1,beta,bias_g,1)
      cublasSgemm(cbHandle,CUBLAS_OP_N,CUBLAS_OP_N,inputs,batchSize,outputs,alpha,weight_d,inputs,actDiff,outputs,beta,srcDiff,inputs)
    }
  }

  def updateWeight(): Unit ={
    val alpha = pointerTo((-1.0f) * lr.toFloat)
    cublasSaxpy(cbHandle,weights.length,alpha,weight_g,1,weight_d,1)
    cublasSaxpy(cbHandle,bias.length,alpha,bias_g,1,bias_d,1)
  }


  def destroy(): Unit ={
    cudaFree(weight_d)
    cudaFree(weight_g)
    cudaFree(bias_d)
    cudaFree(bias_g)
    cudaFree(onevec_d)
    cudaFree(srcDiff)
    cudaFree(actDiff)
    cudaFree(fcD)
    cudaFree(fcActD)
  }
}
