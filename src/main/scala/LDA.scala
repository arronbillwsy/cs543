import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vector => MLVector}

object LDA {

  def lda_model(inputFrame: DataFrame): Unit ={
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setMinDF(4)
      .fit(inputFrame)

    val documents = cvModel.transform(inputFrame).select("rawFeatures").rdd.map {
      case Row(features: MLVector) => Vectors.fromML(features)
    }.zipWithIndex().map(_.swap)
    val vocab = cvModel.vocabulary
    val count = documents.map(_._2.numActives).sum().toLong
    val actualCorpusSize = documents.count()
    val actualVocabSize = vocab.length
    val lda = new LDA()
    lda.setK(1000)
      .setMaxIterations(10)
    val ldaModel = lda.run(documents)
    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 20)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocab(term.toInt), weight) }
    }
    println(s"${5} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
      }
      println()
    }
  }








}
