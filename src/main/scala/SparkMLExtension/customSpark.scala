package SparkMLExtension
import org.apache.spark.ml.{Estimator, Model, Pipeline, Transformer}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark

import scala.util.parsing.json.{JSON, _}


/**
  * Created by daryazmachynskaya on 6/21/17.
  */


//   Use these as guidelines to create your own Transformer.
//     This should take a DataFrame with a single column named value created by sqlContext.read.textFile(<tweetfile>).
//     It should return a DataFrame with three columns: the original value and a tweet and lang column


object customSpark {
  def main(args: Array[String]) {
    val sc = SparkMLExtension.createSpark.main(args) // reuses code to create a SparkContext
    val sqlContext = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this // gives me the .toDF() method to turn an RDD into a DataFrame
    import sqlContext.implicits._
//    import spark.implicits_
    val df = sqlContext.read.textFile("/Users/daryazmachynskaya/gU/Scala-Spark/Tweets")

    val clean_tweet = new customLang()
      .setInputCol("value")

    val index_lang = new LangIndexer()
      .setInputCol("lang")
      .setOutputCol("num_lang")   //we convert the string to a numeric

    val tokenizer = new RegexTokenizer()
      .setInputCol("tweet")
      .setOutputCol("words")
      .setPattern("\\s+|[,.\\\"]")

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(200)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val rf = new RandomForestClassifier()
      .setLabelCol("num_lang")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeline = new Pipeline().setStages(Array(clean_tweet, index_lang, tokenizer, hashingTF, idf, rf))

    val Array(tweets_train, tweets_test) = df.randomSplit(Array(.7,.3))

    val model = pipeline.fit(tweets_train)

    val test_model = model.transform(tweets_test)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("num_lang") //now you have the numeric language as your output
      .setRawPredictionCol("probability")

    println("AUC for Random Forest:", evaluator.setMetricName("areaUnderROC").evaluate(test_model))


  }

  class customLang(override val uid: String) extends Transformer {
    final val inputCol= new Param[String](this, "inputCol", "The input column")
    //  final val outputCol1 = new Param[String](this, "outputCol1", "The first output column")

    def setInputCol(value: String): this.type = set(inputCol, value)

    //  def setOutputCol(value: String): this.type = set(outputCol1, value)

    def this() = this(Identifiable.randomUID("custom_lang"))

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

    def copy(extra: ParamMap): Transformer = {
      defaultCopy(extra)
    }

    override def transformSchema(schema: StructType): StructType = {
      // Check that the input type is a string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField("tweet", StringType, false))
        .add(StructField("lang", StringType, false))
    }

     def transform(df: Dataset[_]): DataFrame = {
      val get_text = udf { in: String => findVal(in, "text") }
      val get_lang = udf { in: String => findVal(in, "lang") }

      df.withColumn("tweet", get_text(col($(inputCol))))
        .withColumn("lang", get_lang(col($(inputCol)))).where("tweet is not null").filter("lang == 'en' or lang == 'es'")

    }
  }

  //After you've finished, let's move on to creating a custom estimator, specifically a simple indexer that will assign an index to each language.

  trait LangIndexerParams extends Params {
    final val inputCol= new Param[String](this, "inputCol", "The input column")
    final val outputCol = new Param[String](this, "outputCol", "The output column")
  }

  class LangIndexer(override val uid: String) extends Estimator[LangIndexerModel] with LangIndexerParams {
//set the input & output column names
    def setInputCol(value: String) = set(inputCol, value)
    def setOutputCol(value: String) = set(outputCol, value)

    def this() = this(Identifiable.randomUID("lang_indexer"))

    //just need to include
    override def copy(extra: ParamMap): LangIndexer = {
      defaultCopy(extra)
    }

    override def transformSchema(schema: StructType): StructType = {
      // Check for string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField($(outputCol), IntegerType, false)) //make sure it returns a numeric
    }

    //Conor's solution for fit
    override def fit(dataset: Dataset[_]): LangIndexerModel = {
      import dataset.sparkSession.implicits._
      val words = dataset.select(dataset($(inputCol)).as[String]).distinct
        .collect()
      val model = new LangIndexerModel(uid, words)
      model.set(inputCol, $(inputCol))
      model.set(outputCol, $(outputCol))
      model
    }

  }

  class LangIndexerModel(override val uid: String, words: Array[String]) extends Model[LangIndexerModel] with LangIndexerParams {

    //we need to re-import findVal because it's not available to this class
    def findVal(str: String, ToFind: String): String = {
      try {
        JSON.parseFull(str) match {
          case Some(m: Map[String, String]) => m(ToFind)
        }
      } catch {
        case e: Exception => null
      }
    }

      override def copy(extra: ParamMap): LangIndexerModel = {
      defaultCopy(extra)
    }


    override def transformSchema(schema: StructType): StructType = {
      // Check that the input type is a string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField($(outputCol), IntegerType, false))
    }

    def mapLang(input: String): Int = {
      if (input == "en") 1
      else if(input == "es") 0
      else -1
    }

    override def transform(dataset: Dataset[_]): DataFrame = {
      val num_lang = udf {in: String => mapLang(in)}

      dataset.select(col("*"), num_lang(dataset.col($(inputCol))).as($(outputCol))) //select everything and rename the column
     }
  }

}



