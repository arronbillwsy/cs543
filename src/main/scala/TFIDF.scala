import breeze.linalg.argtopk
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.{DataFrame, Encoders, SparkSession}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

object TFIDF {



  case class WordsWithID(id: String, words: Array[String])
  case class ArticleFeatures(id: String, features: Vector)

  def tfidf(inputFrame: DataFrame): Unit ={


    val spark = SparkSession
      .builder().master("local")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    implicit val articleEncoder = Encoders.product[ArticleFeatures]
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setMinDF(4)
      .fit(inputFrame)
    val vocab = cvModel.vocabulary
    val featurizedData = cvModel.transform(inputFrame)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel
      .transform(featurizedData)
      .select("id", "features").as[ArticleFeatures]
    rescaledData.take(10).foreach { ele =>
      val breezeVector = toBreeze(ele.features)
      val topKWords = argtopk(breezeVector, 10)
      inputFrame.select("title").show(1)
      println(topKWords.map(vocab(_)).mkString(" "))
      topKWords
    }
  }

  def toBreeze(v: Vector): BV[Double] = v match {
    case DenseVector(values) => new BDV[Double](values)
    case SparseVector(size, indices, values) => {
      new BSV[Double](indices, values, size)
    }
  }

  def toSpark(v: BV[Double]) = v match {
    case v: BDV[Double] => new DenseVector(v.toArray)
    case v: BSV[Double] => new SparseVector(v.length, v.index, v.data)
  }
}
