/**
  * Created by daryazmachynskaya on 6/19/17.
  */
//import SparkMLExtension.createSpark

object MVP {
  def main(args: Array[String]) = {
    val sc = SparkMLExtension.createSpark.main(args)
    println(sc.parallelize(1 to 100).sum)
  }
}
