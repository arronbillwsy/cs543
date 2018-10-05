import breeze.linalg.argtopk
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, SparkSession}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

object getKeywords {

  def main(args: Array[String]): Unit = {
    var filepath = "/C:/Users/31476/Desktop/543/bytecup2018/processed_train.0.txt"
    val data = readWords(filepath)
//    TFIDF.tfidf_model(data)
    LDA.lda_model(data)


  }

  def readWords(path : String) ={
    val spark = SparkSession.builder().master("local").appName("readWords")
      .config("spark.some.config.option", "some-value").getOrCreate()
    import spark.implicits._
    val stringFrame = spark.read.text(path).as[String]
    val jsonFrame = spark.read.json(stringFrame)
    val data = jsonFrame.select("id","words", "title","content").filter("content != ''")
    data
  }








}
