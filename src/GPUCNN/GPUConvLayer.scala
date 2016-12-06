package GPUCNN

import jcuda.{Pointer, Sizeof}
import jcuda.jcudnn.JCudnn._
import jcuda.jcudnn.cudnnActivationMode._
import jcuda.jcudnn._
import jcuda.runtime.JCuda._
import jcuda.jcudnn.cudnnTensorFormat._
import jcuda.jcudnn.cudnnDataType._
import jcuda.jcudnn.cudnnConvolutionMode._
import jcuda.jcudnn.cudnnNanPropagation._
import jcuda.jcudnn.cudnnConvolutionFwdAlgo._
import jcuda.jcudnn.cudnnConvolutionBwdDataAlgo._
import jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.jcublas.JCublas2._
import jcuda.jcublas.cublasHandle


/**import jcuda.jcudnn.cudnnConvolutionFwdPreference._

  * Created by philip on 10/29/16.
  */
class GPUConvLayer (var in_c: Int, var out_c: Int, var k_dim: Int, val in_w: Int, val in_h: Int, val bs:Int) extends utility with Serializable {
  //structure
  val in_channels:Int = in_c
  val out_channels:Int = out_c
  val kernel_dim:Int = k_dim
  val in_width:Int = in_w
  val in_height:Int = in_h
  val out_width:Int = in_width - kernel_dim +1
  val out_height:Int = in_height - kernel_dim + 1
  val batchSize:Int = bs
  var lr = 0.01

  var weights: Array[Float] = Array.fill[Float](in_channels*kernel_dim*kernel_dim*out_channels){(scala.util.Random.nextFloat()*2-1)/10}   //weight
  //var weights: Array[Float] = Array.fill[Float](in_channels*kernel_dim*kernel_dim*out_channels){1f}   //weight
  var bias: Array[Float] = Array.fill[Float](out_channels){(scala.util.Random.nextFloat()*2-1)/10}     //bias
  //var bias: Array[Float] = Array.fill[Float](out_channels){0.1f}     //bias
  var weight_d: Pointer = createDevicePointer(weights)                //weight data pointer
  var bias_d: Pointer = createDevicePointer(bias)                     //bias data pointer
  var weight_g:Pointer = createDevicePointer(weights)
  var bias_g:Pointer = createDevicePointer(bias)

  cudaMalloc(weight_d,weights.length*Sizeof.FLOAT)
  cudaMalloc(bias_d,bias.length*Sizeof.FLOAT)
  cudaMalloc(weight_g,weights.length*Sizeof.FLOAT)
  cudaMalloc(bias_g,bias.length*Sizeof.FLOAT)

  cudaMemcpy(weight_d,Pointer.to(weights),weights.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)
  cudaMemcpy(bias_d,Pointer.to(bias),bias.length*Sizeof.FLOAT,cudaMemcpyHostToDevice)

  var cuHandle: cudnnHandle = _
  var cbHandle: cublasHandle = _

  //input for calculation
  var srcT: cudnnTensorDescriptor = _   //input tensor
  var srcD: Pointer = _                 //input data pointer
  var srcDim: Array[Float] = Array.fill[Float](batchSize*in_channels*in_width*in_height){0f}
  var srcDiff: Pointer = createDevicePointer(srcDim)
  cudaMalloc(srcDiff,srcDim.length*Sizeof.FLOAT)

  //output
  var convT: cudnnTensorDescriptor = new cudnnTensorDescriptor //dst
  cudnnCreateTensorDescriptor(convT)
  var out_vec: Array[Float] = Array.fill[Float](batchSize*out_channels*out_height*out_width){0f}
  var convD:Pointer = createDevicePointer(out_vec)                 //output data pointer
  cudaMalloc(convD,out_vec.length*Sizeof.FLOAT)

  //bias
  val convBiasT: cudnnTensorDescriptor = new cudnnTensorDescriptor
  cudnnCreateTensorDescriptor(convBiasT)
  cudnnSetTensor4dDescriptor(convBiasT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,out_channels,1,1)

  //activation
  var actT: cudnnTensorDescriptor = new cudnnTensorDescriptor
  cudnnCreateTensorDescriptor(actT)

  //filter
  val convfilterDesc = new cudnnFilterDescriptor
  cudnnCreateFilterDescriptor(convfilterDesc)
  //cudnnSetFilterNdDescriptor(convfilterDesc,CUDNN_DATA_FLOAT,4,Array(out_channels,in_channels,kernel_dim,kernel_dim))
  cudnnSetFilter4dDescriptor(convfilterDesc,CUDNN_DATA_FLOAT,out_channels,in_channels,kernel_dim,kernel_dim)

  //convolution
  val convDesc = new cudnnConvolutionDescriptor
  cudnnCreateConvolutionDescriptor(convDesc)
  cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,CUDNN_CONVOLUTION)

  val convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM

  //Layer input for delta
  var nextLayerD_d: Pointer = _
  /*var thisLayerD: Array[Float]=Array.fill[Float](batchSize*out_channels*out_width*out_height){0f}
  var thisLayerD_d: Pointer = createDevicePointer(thisLayerD)   //pass to next layer
  cudaMalloc(thisLayerD_d, thisLayerD.length*Sizeof.FLOAT)
*/
  //activation
  val activation = CUDNN_ACTIVATION_SIGMOID
  val actDesc = new cudnnActivationDescriptor
  cudnnCreateActivationDescriptor(actDesc)
  cudnnSetActivationDescriptor(actDesc,activation,CUDNN_PROPAGATE_NAN,0.0)

  //workspace
  var workSpace: Pointer = _            //workspace pointer
  var workSpaceSize: Long = _             //workspace Size

  def init(ch:cudnnHandle, cb:cublasHandle, srcData:Pointer, srcTensor:cudnnTensorDescriptor, nlD:Pointer): Unit ={
    this.cuHandle = ch
    this.cbHandle = cb
    this.srcD = srcData
    this.srcT = srcTensor
    this nextLayerD_d = nlD

    //output tensor dimension
    val tensorDims: Int = 4
    var tensorOuputDimA: Array[Int] = Array(batchSize,in_channels,in_height,in_width)
    cudnnGetConvolutionNdForwardOutputDim(convDesc, srcT, convfilterDesc, tensorDims, tensorOuputDimA)
    cudnnSetTensor4dDescriptor(convT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,tensorOuputDimA(0),tensorOuputDimA(1),tensorOuputDimA(2),tensorOuputDimA(3))
    cudnnSetTensor4dDescriptor(actT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,tensorOuputDimA(0),tensorOuputDimA(1),tensorOuputDimA(2),tensorOuputDimA(3))

    //workspace
    val sizeInBytesArray: Array[Long] = Array(0)
    workSpace = new Pointer
    cudnnGetConvolutionForwardWorkspaceSize(ch,srcT,convfilterDesc,convDesc,convT,0,sizeInBytesArray)
    workSpaceSize = sizeInBytesArray(0)
    if (workSpaceSize != 0) {
      cudaMalloc(workSpace, workSpaceSize)
    }
  }

  def forwardPropagation(): Unit ={
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
    cudnnConvolutionForward(cuHandle,alpha,srcT,srcD,convfilterDesc,weight_d,convDesc,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,workSpace,workSpaceSize,beta,convT,convD)
    cudnnAddTensor(cuHandle, alpha, convBiasT, bias_d, alpha, convT, convD)
    //cudnnActivationForward(cuHandle,CUDNN_ACTIVATION_SIGMOID,alpha,convT,convD,beta,actT,convD)
  }

  def backwardPropagation(): Unit ={
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
    cudnnConvolutionBackwardBias(cuHandle,alpha,convT,nextLayerD_d,beta,convBiasT,bias_g)
    cudnnConvolutionBackwardFilter(cuHandle,alpha,srcT,srcD,convT,nextLayerD_d,convDesc,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,workSpace,workSpaceSize,beta,convfilterDesc,weight_g)
    cudnnConvolutionBackwardData(cuHandle,alpha,convfilterDesc,weight_d,convT,nextLayerD_d,convDesc,CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,workSpace,workSpaceSize,beta,srcT,srcDiff)
  }

  def updateWeight(): Unit ={
    val alpha: Pointer = pointerTo(-1f*lr.toFloat)
    cublasSaxpy(cbHandle,weights.length,alpha,weight_g,1,weight_d,1)
    cublasSaxpy(cbHandle,bias.length,alpha,bias_g,1,bias_d,1)
  }

  def destroy() {
    cudaFree(weight_d)
    cudaFree(bias_d)
    cudaFree(weight_g)
    cudaFree(bias_g)
    cudaFree(srcDiff)
    cudaFree(convD)
    cudaFree(workSpace)
  }
}
