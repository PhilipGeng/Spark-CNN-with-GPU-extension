package JCudaTest

/**
  * Created by philip on 10/7/16.
  */

import jcuda.driver.CUdevice_attribute._
import jcuda.driver.JCudaDriver._
import java.util._
import jcuda.driver._

object JCudaDeviceQuery {
  /**
    * Entry point of this program
    *
    * @param args Not used
    */
  def main(args:Array[String]): Unit = {
    JCudaDriver.setExceptionsEnabled(true)
    cuInit(0)

    // Obtain the number of devices
    val deviceCountArray: Array[Int] = Array(0)
    cuDeviceGetCount(deviceCountArray)
    val deviceCount: Int = deviceCountArray(0)
    System.out.println("Found " + deviceCount + " devices")

    var i: Int = 0
    while (i < deviceCount) {
      {
        val device: CUdevice = new CUdevice
        cuDeviceGet(device, i)
        // Obtain the device name
        val deviceName: Array[Byte] = new Array[Byte](1024)
        cuDeviceGetName(deviceName, deviceName.length, device)
        val name: String = createString(deviceName)
        // Obtain the compute capability
        val majorArray: Array[Int] = Array(0)
        val minorArray: Array[Int] = Array(0)
        cuDeviceComputeCapability(majorArray, minorArray, device)
        val major: Int = majorArray(0)
        val minor: Int = minorArray(0)
        System.out.println("Device " + i + ": " + name + " with Compute Capability " + major + "." + minor)
        // Obtain the device attributes
        val array: Array[Int] = Array(0)
        val attributes: List[Integer] = getAttributes
        import scala.collection.JavaConversions._
        for (attribute <- attributes) {
          val description: String = getAttributeDescription(attribute)
          cuDeviceGetAttribute(array, attribute, device)
          val value: Int = array(0)
          System.out.printf("    "+description+" : "+value+"\n")
        }
      }
      {
        i += 1; i - 1
      }
    }
  }

  /**
    * Returns a short description of the given CUdevice_attribute constant
    *
    * @param attribute The CUdevice_attribute constant
    * @return A short description of the given constant
    */
  private def getAttributeDescription(attribute: Int): String = {
    attribute match {
      case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK =>
        return "Maximum number of threads per block"
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X =>
        return "Maximum x-dimension of a block"
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y =>
        return "Maximum y-dimension of a block"
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z =>
        return "Maximum z-dimension of a block"
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X =>
        return "Maximum x-dimension of a grid"
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y =>
        return "Maximum y-dimension of a grid"
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z =>
        return "Maximum z-dimension of a grid"
      case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK =>
        return "Maximum shared memory per thread block in bytes"
      case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY =>
        return "Total constant memory on the device in bytes"
      case CU_DEVICE_ATTRIBUTE_WARP_SIZE =>
        return "Warp size in threads"
      case CU_DEVICE_ATTRIBUTE_MAX_PITCH =>
        return "Maximum pitch in bytes allowed for memory copies"
      case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK =>
        return "Maximum number of 32-bit registers per thread block"
      case CU_DEVICE_ATTRIBUTE_CLOCK_RATE =>
        return "Clock frequency in kilohertz"
      case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT =>
        return "Alignment requirement"
      case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT =>
        return "Number of multiprocessors on the device"
      case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT =>
        return "Whether there is a run time limit on kernels"
      case CU_DEVICE_ATTRIBUTE_INTEGRATED =>
        return "Device is integrated with host memory"
      case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY =>
        return "Device can map host memory into CUDA address space"
      case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE =>
        return "Compute mode"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH =>
        return "Maximum 1D texture width"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH =>
        return "Maximum 2D texture width"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT =>
        return "Maximum 2D texture height"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH =>
        return "Maximum 3D texture width"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT =>
        return "Maximum 3D texture height"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH =>
        return "Maximum 3D texture depth"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH =>
        return "Maximum 2D layered texture width"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT =>
        return "Maximum 2D layered texture height"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS =>
        return "Maximum layers in a 2D layered texture"
      case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT =>
        return "Alignment requirement for surfaces"
      case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS =>
        return "Device can execute multiple kernels concurrently"
      case CU_DEVICE_ATTRIBUTE_ECC_ENABLED =>
        return "Device has ECC support enabled"
      case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID =>
        return "PCI bus ID of the device"
      case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID =>
        return "PCI device ID of the device"
      case CU_DEVICE_ATTRIBUTE_TCC_DRIVER =>
        return "Device is using TCC driver model"
      case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE =>
        return "Peak memory clock frequency in kilohertz"
      case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH =>
        return "Global memory bus width in bits"
      case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE =>
        return "Size of L2 cache in bytes"
      case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR =>
        return "Maximum resident threads per multiprocessor"
      case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT =>
        return "Number of asynchronous engines"
      case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING =>
        return "Device shares a unified address space with the host"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH =>
        return "Maximum 1D layered texture width"
      case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS =>
        return "Maximum layers in a 1D layered texture"
      case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID =>
        return "PCI domain ID of the device"
    }
    "(UNKNOWN ATTRIBUTE)"
  }

  /**
    * Returns a list of all CUdevice_attribute constants
    *
    * @return A list of all CUdevice_attribute constants
    */
  private def getAttributes: List[Integer] = {
    val list: List[Integer] = new ArrayList[Integer]
    list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    list.add(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    list.add(CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_PITCH)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    list.add(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
    list.add(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    list.add(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
    list.add(CU_DEVICE_ATTRIBUTE_INTEGRATED)
    list.add(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)
    list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS)
    list.add(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT)
    list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    list.add(CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
    list.add(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    list.add(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    list.add(CU_DEVICE_ATTRIBUTE_TCC_DRIVER)
    list.add(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    list.add(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)
    list.add(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)
    list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS)
    list.add(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
    list
  }

  /**
    * Creates a String from a zero-terminated string in a byte array
    *
    * @param bytes
    * The byte array
    * @return The String
    */
  private def createString(bytes: Array[Byte]): String = {
    val sb: StringBuilder = new StringBuilder
    var i: Int = 0
    var f = true
    while (i < bytes.length && f) {
      {
        val c: Char = bytes(i).toChar
        if (c == 0) {
          f=false
        }
        else{
          sb.append(c)
        }
      }
      {
        i += 1; i - 1
      }
    }
    sb.toString
  }
}
