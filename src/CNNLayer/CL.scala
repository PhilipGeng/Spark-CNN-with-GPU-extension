package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This convolutional layer in CNN
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import org.apache.spark.rdd.RDD
import breeze.numerics.{atan, tanh, sigmoid}
/**
 * constructor:
 * @param numfm number of output feature maps
 * @param dim_conv width/height of perception field/convolutional musk
 */
class CL(val numfm:Int, val dim_conv:Int) extends layer with layerFunc {
  //fm_input_map: by default, input image and feature maps are one-to-one aligned and mapped
  var fm_input_map: Array[Array[Int]] = (0 to numfm-1).map(x=>Array(x)).toArray
  val in_channel = fm_input_map.flatten.max+1
  val out_channel = numfm
  //outputIndex: a invert index for the position of each input map
  var outputIndex: Array[Array[(Int,Int)]] = calcInvIndex(fm_input_map)
  //convolve kernel, learnable weights, init:(-1,1)
  var kernel: Array[Array[DM[Double]]] = fm_input_map.map(arr=>arr.map(fm=>DM.rand[Double](dim_conv,dim_conv):*=2d:-=1d))
  //bias, learnable, init:(-1,1)
  var bias: Array[Double] = (DV.rand(numfm):*=2d:-=1d).toArray
  //[cluster]rdd
  var input: RDD[Array[DM[Double]]] = _
  var delta: RDD[Array[DM[Double]]] = _
  var output: RDD[Array[DM[Double]]]= _
  //[local]
  var inputLocal: Array[DM[Double]] = _
  var deltaLocal: Array[DM[Double]] = _
  var outputLocal: Array[DM[Double]]= _
  var kadj: Array[Array[DM[Double]]] = fm_input_map.map(arr=>arr.map(fm=>DM.fill[Double](dim_conv,dim_conv){0d}))
  var badj: Array[Double] = (DV.fill(numfm){0d}).toArray

  /**
   * set map relationship between input maps with output feature maps
   * @param map input image/map of the ith output feature map
   */

  def set_fm_input_map(map:Array[Array[Int]]): Unit ={
    fm_input_map = map
    outputIndex = calcInvIndex(map)
    kernel = fm_input_map.map(arr=>arr.map(fm=>DM.rand[Double](dim_conv,dim_conv):*=2d:-=1d))
    kadj = fm_input_map.map(arr=>arr.map(fm=>DM.fill[Double](dim_conv,dim_conv){0d}))
  }

  /**
   * calculate position of each input map
   * @param input_map fm_input_map
   * @return each input[(outputfm, index in kernel)]
   */
  def calcInvIndex(input_map:Array[Array[Int]]): Array[Array[(Int,Int)]] ={
    val numOfInputMap:Int = input_map.flatMap(x=>x).max+1
    val ii:Array[Array[(Int,Int)]] = Array.fill(numOfInputMap){Array()}
    for(i<-input_map.indices)
      for(j<-input_map(i).indices)
        ii(input_map(i)(j)) = ii(input_map(i)(j))++Array((i,j))
    ii
  }

  /**
   *  [cluster] batch processing in rdd
   * @param input input Matrix of image or maps
   *              will be distributed to different featuremaps following rules defined in fm_input_map
   * @return output feature maps in Array of matrix
   */

  def forward(input: RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] = {
    this.input = input
    this.input.cache()
    output = input.map{rddarr=>forwardLocal(rddarr)}//rdd end
    output.cache()
    output
  }
  /**
   * [local]
   * @param input input Matrix of image or maps
   *              will be distributed to different featuremaps observing rules defined in fm_input_map
   * @return output feature maps in Array of matrix
   */
  def forwardLocal(input: Array[DM[Double]]): Array[DM[Double]] = {
    this.inputLocal = input
    outputLocal = kernel.zip(bias).zip(fm_input_map).map{fm=> //each fm
      val k: Array[DM[Double]] = fm._1._1 //kernel
      val b: Double = fm._1._2 //bias
      val i: Array[Int] = fm._2 //input map
      val c = k.zip(i).map{ker=> //each convoltuion
        val convKernel: DM[Double] = ker._1 //kernel
        val index: Int = ker._2 //index of inputmap
        convolveDM(inputLocal(index),convKernel) //get inputmap and convolve
      }.reduce((x,y)=>x+y)
      activate(c:+b)
    }//end fm
    outputLocal
  }

  /**
   * distribute task according to type of layers
   * @param nextLayer
   */
  def calErr(nextLayer:FL): Unit ={
    //get data from next layer
    val nextDelta: RDD[DV[Double]] = nextLayer.delta
    val nextWeight: DM[Double] = nextLayer.weight
    //back prop delta by weight matrix to this CL
    val sum:RDD[DV[Double]]= nextDelta.map{vec=>nextWeight*vec}
    //flatten output Array[DM[Double]] to DV[Double]
    val o:RDD[DV[Double]] = output.map(flattenOutput(_))
    //calculate delta of this CL based on output value
    delta = o.zip(sum).map{elem=> // each rdd
      val e_o:DV[Double] = elem._1
      val e_sum:DV[Double] = elem._2
      formatOutput(act_derivative(e_o) :* e_sum)
    }//end rdd
    delta.cache()
  }

   def calErrLocal(nextLayer:FL): Unit = {
    //get data from next layer
    val nextDelta: DV[Double] = nextLayer.deltaLocal
    val nextWeight: DM[Double] = nextLayer.weight
    //back prop delta by weight matrix to this CL
    val sum:DV[Double]= nextWeight*nextDelta
    //flatten output Array[DM[Double]] to DV[Double]
    val o:DV[Double] = flattenOutput(outputLocal)
    //calculate delta of this CL based on output value
    val deltaVector:DV[Double] = act_derivative(o) :* sum
    deltaLocal = formatOutput(deltaVector)
  }

  def calErr(nextLayer:SL): Unit = {
    val SLDelta: RDD[Array[DM[Double]]] = nextLayer.delta
    val SLWeight: Array[Double] = nextLayer.weight
    val SLDim: Int = nextLayer.dim_neighbor
    val propErr: RDD[Array[DM[Double]]] = SLDelta.map{arr=> //each arr
      arr.zip(SLWeight).map{Mat=> //each mat zip with weight
        val d:DM[Double] = Mat._1 //delta mat
        val w:Double = Mat._2 //weight
        expandDM(d,SLDim):*w
      }
    }
    //distribute delta to featuremaps
    delta = propErr.zip(output).map{arr=>
      arr._1.zip(arr._2).map{item=>
        val d:DM[Double] = item._1
        val o:DM[Double] = item._2
        act_derivative(o):*d
      }
    }
    delta.cache()
  }

  def calErrLocal(nextLayer:SL): Unit = {
    //ask next SL to calculate delta for this CL
    val SLDeltaLocal = nextLayer.deltaLocal
    val SLWeight = nextLayer.weight
    val SLDim = nextLayer.dim_neighbor
    val propErr: Array[DM[Double]] = SLDeltaLocal.zip(SLWeight).map{Mat=>
      val d:DM[Double] = Mat._1 //delta mat
      val w:Double = Mat._2 //weight
      expandDM(d,SLDim):*w
    }
    deltaLocal = propErr.zip(outputLocal).map{each=>
      val d = each._1 //delta
      val o = each._2 //output
      act_derivative(o):*d
    }
  }

  /**
   * [cluster]
   * adjust weight
   */
  def adjWeight(): Unit ={
    val adj:Array[(Array[DM[Double]],Double)] =delta.zip(input).map{rddarr=> //each iteration
      val err:Array[DM[Double]] = rddarr._1 //delta
      val input:Array[DM[Double]] = rddarr._2 //input
      err.zip(fm_input_map.map(x=>x.map(input(_)))).map{d=> //each delta or each kernel
        val delta:DM[Double] = d._1 //delta
        val input:Array[DM[Double]] = d._2 //input
        val adjb:Double = sum(delta) //bias
        val adjw:Array[DM[Double]] = input.map{in=>convolveDM(in,delta)} //adj Kernel
        (adjw,adjb)
      }
    }.reduce{case(a,b)=>
      a.zip(b).map{pair=>(
          pair._1._1.zip(pair._2._1).map(mat=>mat._1+mat._2), // sum adjw matrix
          pair._1._2+pair._2._2 //sum bias double
        )
      }
    }
    for(i<-adj.indices){
      val adjw:Array[DM[Double]] = adj(i)._1
      badj(i) = adj(i)._2*eta+badj(i)*momentum
      bias(i) = bias(i)+badj(i)
      for(j<-adjw.indices){
        require(kernel(i)(j).rows == adjw(j).rows)
        require(kernel(i)(j).cols == adjw(j).cols)
        kadj(i)(j) = (adjw(j):*eta)+(kadj(i)(j):*momentum)
        kernel(i)(j) = kernel(i)(j) + kadj(i)(j)
      }
    }
  }
  /**
   * [local]
   * adjust weight
   */
  def adjWeightLocal(): Unit ={
    val adj:Array[(Array[DM[Double]],Double)] = deltaLocal.zip(fm_input_map.map(x=>x.map(inputLocal(_)))).map{d=>
      val delta:DM[Double] = d._1
      val input:Array[DM[Double]] = d._2
      val adjb:Double = sum(delta)
      val adjw:Array[DM[Double]] = input.map{in=>convolveDM(in,delta)}
      (adjw,adjb)
    }
    for(i<-adj.indices){
      val adjw:Array[DM[Double]] = adj(i)._1
      badj(i) = adj(i)._2*eta+momentum*badj(i)
      bias(i) = bias(i)+badj(i)
      for(j<-adjw.indices){
        require(kernel(i)(j).rows == adjw(j).rows)
        require(kernel(i)(j).cols == adjw(j).cols)
        kadj(i)(j) = (adjw(j):*eta)+(kadj(i)(j):*momentum)
        kernel(i)(j) = kernel(i)(j) + kadj(i)(j)
      }
    }
  }

  def clearCache(): Unit ={
    input.unpersist()
    output.unpersist()
    delta.unpersist()
  }
  def calErr(n:CL):Unit = {}
  def calErrLocal(n:CL):Unit = {}
  def calErr(target: RDD[DV[Double]]): Unit={}
  def calErrLocal(target: DV[Double]): Unit={}

}
