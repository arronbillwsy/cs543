import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.DistributedLDAModel

object LDA {

  def lda_model(inputFrame: DataFrame): Unit ={
    val vectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
    val numTopics = 10
    val lda = new LDA()
      .setK(numTopics)
      .setMaxIter(50)
      .setOptimizer("em")
    val pipeline = new Pipeline().setStages(Array(vectorizer, lda))
    val pipelineModel = pipeline.fit(inputFrame)
    val vectorizerModel = pipelineModel.stages(0).asInstanceOf[CountVectorizerModel]
    val ldaModel = pipelineModel.stages(1).asInstanceOf[DistributedLDAModel]
    ldaModel.trainingLogLikelihood
    val vocabList = vectorizerModel.vocabulary
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }
    val topics = ldaModel.describeTopics(maxTermsPerTopic = 5)
      .withColumn("terms", termsIdx2Str(col("termIndices")))
    val res = topics.select("topic", "terms", "termWeights")
    val zipUDF = udf { (terms: Seq[String], probabilities: Seq[Double]) => terms.zip(probabilities) }
    val topicsTmp = topics.withColumn("termWithProb", explode(zipUDF(col("terms"), col("termWeights"))))
    val termDF = topicsTmp.select(
      col("topic").as("topicId"),
      col("termWithProb._1").as("term"),
      col("termWithProb._2").as("probability"))

  }








}
