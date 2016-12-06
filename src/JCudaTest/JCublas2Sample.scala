package JCudaTest

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */

import jcuda.jcublas.JCublas2._
import jcuda.jcublas.cublasOperation.CUBLAS_OP_N
import jcuda.runtime.JCuda._
import java.util.Random
import jcuda._
import jcuda.jcublas.cublasHandle

/**
  * This is a sample class demonstrating the application of JCublas2 for
  * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
  * C = alpha * A * B + beta * C <br />
  * for single-precision floating point values alpha and beta, and matrices A, B
  * and C of size 1000x1000.
  */
object JCublas2Sample {
  def main(args: Array[String]) {
    testSgemm(1000)
  }

  /**
    * Test the JCublas sgemm operation for matrices of size n x x
    *
    * @param n The matrix size
    */
  def testSgemm(n: Int) {
    val alpha: Float = 0.3f
    val beta: Float = 0.7f
    val nn: Int = n * n
    System.out.println("Creating input data...")
    val h_A: Array[Float] = createRandomFloatData(nn)
    val h_B: Array[Float] = createRandomFloatData(nn)
    val h_C: Array[Float] = createRandomFloatData(nn)
    val h_C_ref: Array[Float] = h_C.clone
    System.out.println("Performing Sgemm with Java...")
    sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref)
    System.out.println("Performing Sgemm with JCublas...")
    sgemmJCublas(n, alpha, h_A, h_B, beta, h_C)
    val passed: Boolean = isCorrectResult(h_C, h_C_ref)
    System.out.println("testSgemm " + (if (passed) "PASSED"
    else "FAILED"))
  }

  /**
    * Implementation of sgemm using JCublas
    */
  private def sgemmJCublas(n: Int, alpha: Float, A: Array[Float], B: Array[Float], beta: Float, C: Array[Float]) {
    val nn: Int = n * n
    // Create a CUBLAS handle
    val handle: cublasHandle = new cublasHandle
    cublasCreate(handle)
    // Allocate memory on the device
    val d_A: Pointer = new Pointer
    val d_B: Pointer = new Pointer
    val d_C: Pointer = new Pointer
    cudaMalloc(d_A, nn * Sizeof.FLOAT)
    cudaMalloc(d_B, nn * Sizeof.FLOAT)
    cudaMalloc(d_C, nn * Sizeof.FLOAT)
    // Copy the memory from the host to the device
    cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1)
    cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1)
    cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1)
    // Execute sgemm
    val pAlpha: Pointer = Pointer.to(Array[Float](alpha))
    val pBeta: Pointer = Pointer.to(Array[Float](beta))
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, pAlpha, d_A, n, d_B, n, pBeta, d_C, n)
    // Copy the result from the device to the host
    cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1)
    // Clean up
    cudaFree(d_A)
    cudaFree(d_B)
    cudaFree(d_C)
    cublasDestroy(handle)
  }

  /**
    * Simple implementation of sgemm, using plain Java
    */
  private def sgemmJava(n: Int, alpha: Float, A: Array[Float], B: Array[Float], beta: Float, C: Array[Float]) {
    var i: Int = 0
    while (i < n) {
      {
        var j: Int = 0
        while (j < n) {
          {
            var prod: Float = 0
            var k: Int = 0
            while (k < n) {
              {
                prod += A(k * n + i) * B(j * n + k)
              }
              {
                k += 1; k
              }
            }
            C(j * n + i) = alpha * prod + beta * C(j * n + i)
          }
          {
            j += 1; j
          }
        }
      }
      {
        i += 1; i
      }
    }
  }

  /**
    * Creates an array of the specified size, containing some random data
    */
  private def createRandomFloatData(n: Int): Array[Float] = {
    val random: Random = new Random
    val x: Array[Float] = new Array[Float](n)
    var i: Int = 0
    while (i < n) {
      {
        x(i) = random.nextFloat
      }
      {
        i += 1; i - 1
      }
    }
    return x
  }

  /**
    * Compares the given result against a reference, and returns whether the
    * error norm is below a small epsilon threshold
    */
  private def isCorrectResult(result: Array[Float], reference: Array[Float]): Boolean = {
    var errorNorm: Float = 0
    var refNorm: Float = 0
    var i: Int = 0
    while (i < result.length) {
      {
        val diff: Float = reference(i) - result(i)
        errorNorm += diff * diff
        refNorm += reference(i) * result(i)
      }
      {
        i += 1; i
      }
    }
    errorNorm = Math.sqrt(errorNorm).toFloat
    refNorm = Math.sqrt(refNorm).toFloat
    if (Math.abs(refNorm) < 1e-6) {
      return false
    }
    return (errorNorm / refNorm < 1e-6f)
  }
}