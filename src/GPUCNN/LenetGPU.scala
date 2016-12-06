package GPUCNN
import java.io._
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

import jcuda.{Pointer, Sizeof}
import jcuda.jcublas.JCublas2.{cublasCreate, cublasDestroy, cublasSgemv}
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.jcudnn.cudnnActivationMode._
import jcuda.jcudnn.cudnnPoolingMode._
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T
import jcuda.jcudnn.JCudnn.{CUDNN_VERSION, cudnnActivationForward, cudnnAddTensor, cudnnConvolutionForward, cudnnCreate, cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor, cudnnCreateLRNDescriptor, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy, cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyLRNDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithm, cudnnGetConvolutionForwardWorkspaceSize, cudnnGetConvolutionNdForwardOutputDim, cudnnGetErrorString, cudnnGetPoolingNdForwardOutputDim, cudnnGetVersion, cudnnLRNCrossChannelForward, cudnnPoolingForward, cudnnSetConvolutionNdDescriptor, cudnnSetFilterNdDescriptor, cudnnSetLRNDescriptor, cudnnSoftmaxForward, _}
import jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU
import jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT
import jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
import jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION
import jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT
import jcuda.jcudnn.cudnnConvolutionMode._
import jcuda.jcudnn.cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1
import jcuda.jcudnn.cudnnPoolingMode._
import jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE
import jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL
import jcuda.jcudnn._
import jcuda.jcudnn.cudnnAddMode._
import jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind.{cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice}

import scala.io.Source

/**
  * Created by philip on 10/26/16.
  */
object LenetGPU extends utility {
  var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
  val dir:String = "/home/philip/Desktop/SparkNNDL-master/mnist_data/"
  var testFile = "mnist_test.csv"
  var lenet = new GPUCNN

  def main(args:Array[String]): Unit ={
    var c = 0
    trainfile = trainfile.map(x=>dir+x)
    testFile = dir + testFile
    var max_epoch = 0
    var t1 = java.lang.System.currentTimeMillis()
    //training phase
    for(i<-0 to max_epoch){
      print(i+": ")
      trainfile.foreach{f=>
        for(line<-Source.fromFile(f).getLines()){
          val splits = line.split(",").map(_.toFloat)
          val l = splits(0).toInt
          val d = splits.drop(1).map(_/255)
          lenet.networkTrain(d,l)
          c=c+1
        }
      }
    }
    var t2 = java.lang.System.currentTimeMillis()
    //testing phase
    println(test())
    println("time consumption: "+(t2-t1))
    lenet.destroy()
  }

  def test(): Float ={
    var total = 0
    var correct = 0
    for(line<-Source.fromFile(testFile).getLines()){
      val splits = line.split(",").map(_.toFloat)
      val l = splits(0).toInt
      val d = splits.drop(1).map(_/255)
      val predicted = lenet.networkTest(d)
      total = total+1
      if(predicted == l)
        correct = correct+1
    }
    correct.toFloat/total.toFloat
  }
}
