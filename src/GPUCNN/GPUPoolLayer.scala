package GPUCNN

import jcuda.jcudnn.JCudnn._
import jcuda.jcudnn.cudnnActivationMode._
import jcuda.jcudnn.{cudnnActivationDescriptor, cudnnHandle, cudnnPoolingDescriptor, cudnnTensorDescriptor}
import jcuda.jcudnn.cudnnPoolingMode._
import jcuda.jcudnn.cudnnNanPropagation._
import jcuda.jcudnn.cudnnTensorFormat._
import jcuda.jcudnn.cudnnDataType._
import jcuda._
import jcuda.runtime.JCuda._

/**
  * Created by philip on 10/29/16.
  */
class GPUPoolLayer(var Size:Int, var Stride:Int, var in_channel_from_previous_conv:Int, var in_height:Int, var in_width:Int, var bs:Int) extends utility with Serializable {
  val size:Int = Size
  val stride:Int = Stride
  val batchSize:Int = bs
  val in_channels:Int = in_channel_from_previous_conv
  val out_channels:Int = in_channels
  val out_height:Int = in_height/stride
  val out_width:Int = in_width/stride
  var lr = 0.01

  var cuHandle: cudnnHandle = _
  //input for calculation
  var srcT: cudnnTensorDescriptor = _   //input tensor
  var srcD: Pointer = _                 //input data pointer
  var srcDim: Array[Float] = Array.fill[Float](batchSize*in_channels*in_width*in_height){0f}
  var srcDiff: Pointer = createDevicePointer(srcDim)
  cudaMalloc(srcDiff,srcDim.length*Sizeof.FLOAT)

  var poolT: cudnnTensorDescriptor = new cudnnTensorDescriptor    //pool Tensor
  cudnnCreateTensorDescriptor(poolT)
  cudnnSetTensor4dDescriptor(poolT,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batchSize,out_channels,out_height,out_width)
  var out_vec: Array[Float] = Array.fill[Float](batchSize*out_channels*out_height*out_width){0f}
  var poolD: Pointer = createDevicePointer(out_vec)                 //output data pointer
  cudaMalloc(poolD,out_vec.length*Sizeof.FLOAT)

  var poolDesc: cudnnPoolingDescriptor = new cudnnPoolingDescriptor
  cudnnCreatePoolingDescriptor(poolDesc)
  cudnnSetPooling2dDescriptor_v4(poolDesc,CUDNN_POOLING_MAX,CUDNN_PROPAGATE_NAN,size,size,0,0,stride,stride)

  //Layer input for delta
  var nextLayerD_d: Pointer = _
  /*var nextLayerD: Array[Float]=Array.fill[Float](batchSize*out_channels*out_width*out_height){0f}
  var nextLayerD_d: Pointer = createDevicePointer(nextLayerD)   //pass to next layer
  cudaMalloc(nextLayerD_d, nextLayerD.length*Sizeof.FLOAT)
*/
  var activation = CUDNN_ACTIVATION_SIGMOID
  val actDesc = new cudnnActivationDescriptor
  cudnnCreateActivationDescriptor(actDesc)
  cudnnSetActivationDescriptor(actDesc,activation,CUDNN_PROPAGATE_NAN,0.0)


  def init(ch:cudnnHandle, srcData:Pointer, srcTensor:cudnnTensorDescriptor, nlD:Pointer): Unit = {
    this.cuHandle = ch
    this.srcD = srcData
    this.srcT = srcTensor
    this nextLayerD_d = nlD
  }

  def forwardPropagation(): Unit = {
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
    cudnnPoolingForward(cuHandle,poolDesc,alpha,srcT,srcD,beta,poolT,poolD)
  }

  def backwardPropagation(): Unit ={
    val alpha: Pointer = pointerTo(1.0f)
    val beta: Pointer = pointerTo(0.0f)
    cudnnPoolingBackward(cuHandle,poolDesc,alpha,poolT,poolD,poolT,nextLayerD_d,srcT,srcD,beta,srcT,srcDiff)
  }

  def destroy(): Unit ={
    cudaFree(srcDiff)
    cudaFree(poolD)
  }
}

