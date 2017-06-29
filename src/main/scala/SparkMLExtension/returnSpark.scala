package SparkMLExtension

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.SQLContext

import scala.util.parsing.json._



/**
  * Created by daryazmachynskaya on 6/19/17.
  */
object SparkTweets {
  def main(args: Array[String]) {
    val sc = SparkMLExtension.createSpark.main(args) // reuses code to create a SparkContext
    val sqlContext = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this // gives me the .toDF() method to turn an RDD into a DataFrame
    import sqlContext.implicits._
    val tweets = sc.textFile("/Users/daryazmachynskaya/gU/Scala-Spark/Tweets")

//    sc.setLogLevel("ERROR")
//
//    //CNTRL + SHIFT + P to see the type
//
    val df = tweets.map(s => getTweetsAndLang(s)).filter(_._1 != "unknown").toDF("tweet","lang")
////
    val tokenizer = new RegexTokenizer()
      .setInputCol("tweet")
      .setOutputCol("words")
      .setPattern("\\s+|[,.\\\"]")
//
    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(200)
////
//    //idf = IDF(inputCol="rawFeatures", outputCol="features")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//
//    //forestizer = RandomForestClassifier(labelCol="lang", featuresCol="features", numTrees=10)
//    //alt + enter
    val rf = new RandomForestClassifier()
      .setLabelCol("lang")
      .setFeaturesCol("features")
      .setNumTrees(10)
////
//    pipeline = Pipeline(stages=[\
//      tokenizer,
//      hashingTF,
//      idf,
//      forestizer])
//
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, rf))
//
////    tweets_train, tweets_test = df.randomSplit([0.7, 0.3], seed=123)
//
    val Array(tweets_train, tweets_test) = df.randomSplit(Array(.7,.3))
//
////    model = pipeline.fit(tweets_train)
    val model = pipeline.fit(tweets_train)
//
////    test_model = model.transform(tweets_test)
    val test_model = model.transform(tweets_test)
//
//
//    //evaluator = BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol='lang')
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("lang")
      .setRawPredictionCol("probability")
//
////    print('AUC for Random Forest:', evaluator.evaluate(test_model{evaluator.metricName: 'areaUnderROC'}))
//
    println("AUC for Random Forest:", evaluator.setMetricName("areaUnderROC").evaluate(test_model))
//
////
////    sc.stop()
//
  }

  def findVal(str: String, ToFind: String): String = {
    try {
      JSON.parseFull(str) match {
        case Some(m: Map[String, String]) => m(ToFind)
      }
    } catch {
      case e: Exception => null
    }
  }


  def getTweetsAndLang(input: String): (String, Int) = {
    try {
      var result = (findVal(input, "text"), -1)

      if (findVal(input, "lang") == "en") result.copy(_2 = 0)
      else if (findVal(input, "lang") == "es") result.copy(_2 = 1)
      else result
    } catch {
      case e: Exception => ("unknown", -1)
    }
  }
}



