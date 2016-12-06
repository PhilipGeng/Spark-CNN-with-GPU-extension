package GPUCNN

import java.io._
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

import jcuda.{Pointer, Sizeof}
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._

/**
  * Created by philip on 10/26/16.
  */
trait utility {
  @throws[IOException]
  def readBinaryFile(fileName: String): Array[Float] = {
    val fis: FileInputStream = new FileInputStream(new File(fileName))
    val data: Array[Byte] = readFully(fis)
    val bb: ByteBuffer = ByteBuffer.wrap(data)
    bb.order(ByteOrder.nativeOrder)
    val fb: FloatBuffer = bb.asFloatBuffer
    val result: Array[Float] = new Array[Float](fb.capacity)
    fb.get(result)
    return result
  }

  def readBinaryFileUnchecked(fileName: String): Array[Float] = {
    try {
      return readBinaryFile(fileName)
    }
    catch {
      case e: IOException => {
        cudaDeviceReset
        e.printStackTrace()
        System.exit(-1)
        return null
      }
    }
  }

  @throws[IOException]
  def readFully(inputStream: InputStream): Array[Byte] = {
    val baos: ByteArrayOutputStream = new ByteArrayOutputStream
    val buffer: Array[Byte] = new Array[Byte](1024)
    var flag = true
    while (flag) {
      {
        val n: Int = inputStream.read(buffer)
        if (n < 0) {
          flag = false //todo: break is not supported
        }
        if(flag)
          baos.write(buffer, 0, n)
      }
    }
    val data: Array[Byte] = baos.toByteArray
    return data
  }

  @SuppressWarnings(Array("deprecation"))
  @throws[IOException]
  def readBinaryPortableGraymap8bitData(inputStream: InputStream): Array[Byte] = {
    val dis: DataInputStream = new DataInputStream(inputStream)
    var line: String = null
    var firstLine: Boolean = true
    var width: Integer = null
    var maxBrightness: Integer = null
    var flag = true
    while (flag) {
      {
        // The DataInputStream#readLine is deprecated,
        // but for ASCII input, it is safe to use it
        line = dis.readLine
        if (line == null) {
          flag=false //todo: break is not supported
        }
        if(flag){
          line = line.trim
          var innerFlag=true
          if (line.startsWith("#")) {
            innerFlag = false //todo: continue is not supported
          }
          if(innerFlag){
            if (firstLine) {
              firstLine = false
              if (line != "P5") {
                throw new IOException("Data is not a binary portable " + "graymap (P5), but " + line)
              }
              else {
                innerFlag = false //todo: continue is not supported
              }
            }
            if(innerFlag){
              if (width == null) {
                val tokens: Array[String] = line.split(" ")
                if (tokens.length < 2) {
                  throw new IOException("Expected dimensions, found " + line)
                }
                width = parseInt(tokens(0))
              }
              else if (maxBrightness == null) {
                maxBrightness = parseInt(line)
                if (maxBrightness > 255) {
                  throw new IOException("Only 8 bit values supported. " + "Maximum value is " + maxBrightness)
                }
                flag=false //todo: break is not supported
              }
            }
          }
        }
      }
    }
    val data: Array[Byte] = readFully(inputStream)
    return data
  }

  @throws[IOException]
  def parseInt(s: String): Integer = {
    try {
      return s.toInt
    }
    catch {
      case e: NumberFormatException => {
        throw new IOException(e)
      }
    }
  }

  @throws[IOException]
  def readImageData(fileName: String): Array[Float] = {
    val is: InputStream = new FileInputStream(new File(fileName))
    val data: Array[Byte] = readBinaryPortableGraymap8bitData(is)
    val imageData: Array[Float] = new Array[Float](data.length)
    var i: Int = 0
    while (i < data.length) {
      {
        imageData(i) = ((data(i).toInt) & 0xff) / 255.0f
      }
      {
        i += 1; i - 1
      }
    }
    return imageData
  }

  def readImageDataUnchecked(fileName: String): Array[Float] = {
    try {
      return readImageData(fileName)
    }
    catch {
      case e: IOException => {
        cudaDeviceReset
        e.printStackTrace()
        System.exit(-1)
        return null
      }
    }
  }

  def createDevicePointer(data: Array[Float]): Pointer = {
    val size: Int = data.length * Sizeof.FLOAT
    val deviceData: Pointer = new Pointer
    cudaMalloc(deviceData, size)
    cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice)
    return deviceData
  }

  def resize(numberOfFloatElements: Int, data: Pointer) {
    cudaFree(data)
    cudaMalloc(data, numberOfFloatElements * Sizeof.FLOAT)
  }

  def pointerTo(value: Float): Pointer = {
    return Pointer.to(Array[Float](value))
  }

  def printDeviceVector(size: Int, d: Pointer) {
    val h: Array[Float] = new Array[Float](size)
    cudaDeviceSynchronize
    cudaMemcpy(Pointer.to(h), d, size * Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    var i: Int = 0
    while (i < size) {
      {
        System.out.print(h(i) + " ")
      }
      {
        i += 1; i - 1
      }
    }
    System.out.println()
  }
}
