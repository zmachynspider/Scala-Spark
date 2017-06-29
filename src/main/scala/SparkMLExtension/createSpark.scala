/**
  * Created by daryazmachynskaya on 6/19/17.
  */
package SparkMLExtension

import org.apache.spark.{SparkConf, SparkContext}

object createSpark {

  def main(args: Array[String]) = {
    val conf = new SparkConf()
      .setAppName("Simple App")
      .setMaster("local[*]")
//      .set("spark.driver.host", "local[*]")
    val sc = new SparkContext(conf)

    sc
  }
}
