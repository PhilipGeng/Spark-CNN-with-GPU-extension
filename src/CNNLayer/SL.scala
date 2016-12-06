package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is the sub-sampling layer in CNN
 */
import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM, sum}
import org.apache.spark.rdd.RDD
import breeze.numerics.{atan, tanh, sigmoid}

/**
 *
 * @param numfm number of featuremaps
 * @param dim_neighbor width/height of the subsampling block
 */
class SL (val numfm: Int, val dim_neighbor:Int) extends layer with layerFunc{
  //weight and bias
  var weight:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  var bias:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  //[cluster]rdd
  var output:RDD[Array[DM[Double]]]=_
  var delta:RDD[Array[DM[Double]]]=_
  var input:RDD[Array[DM[Double]]]=_
  //[local]
  var outputLocal:Array[DM[Double]]=_
  var deltaLocal:Array[DM[Double]]=_
  var inputLocal:Array[DM[Double]]=_
  var wadj:Array[Double] = Array.fill(numfm){0d}
  var badj:Array[Double] = Array.fill(numfm){0d}

  /**
   * forward
   * @param input_arr input image
   * @return output featuremap
   */
  def forward(input_arr:RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] ={
    input = input_arr //save input
    input.cache()
    output = input.map{arr=>forwardLocal(arr)}.cache()
    output
  }

  def forwardLocal(input_arr:Array[DM[Double]]): Array[DM[Double]] ={
    inputLocal = input_arr
    val rm_dim:Int = inputLocal(0).cols/dim_neighbor //output matrix dimension
    outputLocal = inputLocal.zip(weight.zip(bias)).map{mat=> //(input:DM,(weight:Double,bias:Double))
      val dm:DM[Double] = mat._1 //matrix
      val w:Double = mat._2._1 //weight
      val b:Double = mat._2._2 //bias
      DM.tabulate(rm_dim,rm_dim){case (i,j)=> //indices in output matrix
        //mean subsampling
        val mean:Double = sum(dm(i*dim_neighbor to (i+1)*dim_neighbor-1,j*dim_neighbor to (j+1)*dim_neighbor-1))/(dim_neighbor*dim_neighbor)
        activate(mean*w+b)
      }
    }
    outputLocal
  }
  /**
   * calculate Err when it's before CL
   * @param nextLayer
   */
  def calErr(nextLayer:CL): Unit ={
    val CLDelta = nextLayer.delta
    val CLDim = nextLayer.dim_conv
    val CLOutIndex = nextLayer.outputIndex
    val CLWeight = nextLayer.kernel
    val propErr:RDD[Array[DM[Double]]] = CLDelta.map{rddarr=>
      CLOutIndex.map{sm=>
        sm.map { cm =>
          val d = rddarr(cm._1)
          val k = rot180(CLWeight(cm._1)(cm._2))
          val expand_d:DM[Double] = padDM(d,CLDim-1)
          convolveDM(expand_d,k)
        }.reduce((a,b)=>a+b)
      }
    }

    delta = propErr.zip(output).map{arr=> //each arr
      arr._1.zip(arr._2).map{Mat=> //zipped matrix
        val d:DM[Double] = Mat._1 //delta
        val o:DM[Double] = Mat._2 //output
        act_derivative(o):*d //calculate as matrix
      }
    }
    delta.cache()
  }

  def calErrLocal(nextLayer:CL): Unit ={
    val CLDeltaLocal = nextLayer.deltaLocal
    val CLDim = nextLayer.dim_conv
    val CLOutIndex = nextLayer.outputIndex
    val CLWeight = nextLayer.kernel
    val propErr: Array[DM[Double]] = CLOutIndex.map{sm=> //each SL map
      sm.map { cm => //corresponding CL map
        val d = CLDeltaLocal(cm._1) //get CL map delta
        val k = rot180(CLWeight(cm._1)(cm._2)) //rot 180 of conv kernel
        val expand_d:DM[Double] = padDM(d,CLDim-1)
        convolveDM(expand_d,k)
      }.reduce((a,b)=>a+b)
    }

    deltaLocal = outputLocal.zip(propErr).map{x=>
      val o = x._1
      val d = x._2
     act_derivative(o):*d
    }
  }

  /**
   * calculate weight adjustments and queue up in the list
   * cluster mode, for batch update
   */
  def adjWeight(): Unit ={
    val adj:RDD[Array[(Double, Double)]] = delta.zip(input).map{arr => // each arr
      arr._1.zip(arr._2).map { mat=> //each delta mat zip with input mat
        val d:DM[Double] = mat._1 //delta
        val i:DM[Double] = mat._2 //input
        val summat:DM[Double] = DM.tabulate(d.rows,d.cols){case(m,n)=>sum(i(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))}
        val adjw:Double = sum(summat:*d) //weight adjustment
        val adjb:Double = sum(d) //bias adjustment
        (adjw,adjb)
      }
    }
    val redadj:Array[(Double,Double)] = adj.reduce{(a,b)=>
      a.zip(b).map{each=>
        (each._1._1+each._2._1,each._1._2+each._2._2) //reduce(sum up) adjustments
      }
    }
    wadj =redadj.zip(wadj).map(x=>(x._1._1*eta)+(x._2*momentum))
    badj =redadj.zip(badj).map(x=>(x._1._2*eta)+(x._2*momentum))

    weight = weight.zip(wadj).map{arr=>arr._1+arr._2}
    bias = bias.zip(badj).map{arr=>arr._1+arr._2}
  }

  def adjWeightLocal(): Unit ={
    val adjw:Array[Double] = weight.map(x=>0d)
    val adjb:Array[Double] = bias.map(x=>0d)
    for(i<-deltaLocal.indices){
      var adjw_part = 0d
      var adjb_part = 0d
      for(m<-0 to deltaLocal(i).cols-1){
        for(n<-0 to deltaLocal(i).rows-1){
          val sumin = sum(inputLocal(i)(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))
          adjw_part = adjw_part + (sumin*deltaLocal(i)(m,n))
          adjb_part = adjb_part + deltaLocal(i)(m,n)
        }
      }
      wadj(i) = (adjw_part*eta) + (wadj(i)*momentum)
      badj(i) = (adjb_part*eta) + (badj(i)*momentum)
      weight(i) = weight(i) + wadj(i)
      bias(i) = bias(i) + badj(i)
    }
  }
  def clearCache(): Unit ={
    input.unpersist()
    output.unpersist()
    delta.unpersist()
  }
  def calErr(target: RDD[DV[Double]]): Unit={}
  def calErrLocal(target: DV[Double]): Unit={}
  def calErr(nextlayer: FL): Unit = {}
  def calErr(nextlayer: SL): Unit = {}
  def calErrLocal(nextLayer: SL): Unit = {}
  def calErrLocal(nextLayer: FL): Unit = {}
}
