package JCudaTest

import jcuda.Pointer
import jcuda.runtime.JCuda

/**
  * Created by philip on 10/7/16.
  */
object JCudaRuntimeTest {
  def main(args:Array[String]): Unit ={
    val pointer: Pointer = new Pointer
    JCuda.cudaMalloc(pointer, 4)
    System.out.println("Pointer: " + pointer)
    JCuda.cudaFree(pointer)
  }
}
