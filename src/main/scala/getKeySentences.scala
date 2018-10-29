import org.apache.spark.sql.SparkSession

object getKeySentences {
  val spark = SparkSession
    .builder().master("local")
    .appName("Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
  import spark.implicits._

  case class SentenceIDLists(sentence_ids: Array[Long], sentences: Array[Array[String]])

  case class Relation(first: Long, second: Long, weight: Double)

  case class Article(id: String, sentence_list: Array[Array[String]])

  case class ArticleWithScores(id: String, sentence_list: Array[Array[String]], scores: Array[Double])


  def main(args: Array[String]): Unit = {
    val top_k = 10
    val i = 0;
    for( i <- 0 to 8) {
      //      val wordFile = s"file:///C:/Users/31476/Desktop/543/bytecup2018/bytecup.corpus.train.${i}.txt"
      val wordFile = s"file:////media/wsy/DATA/Data/preprocess_data/processed_train.${i}.txt"
//      val outputPath = s"file:////media/wsy/DATA/Data/preprocess_data/processed_key_sen_train.${i}.txt"
      val outputPath = s"file:////home/wsy/桌面/Bytecup2018/key_sen/processed_key_sen_train.${i}.txt"
      var data = readContent(wordFile).select('id,'content.as('sentence_list)).as[Article]

      val articleScoresDataset = data.map{ art =>
        val vertices = art.sentence_list.zipWithIndex
        val similarityMatrix = vertices.map(vertex => vertices
          .map(otherVetex => calculate_score(vertex, otherVetex, 0.5))
        ).map(normalizeArray)
        val neighbours = similarityMatrix.map{arr =>
          arr.zipWithIndex.filter(ele => ele._1 > 0).map(_._2)
        }
        val scores = textRank(similarityMatrix, neighbours, 0.001)
        ArticleWithScores(art.id, art.sentence_list, scores = scores)
      }


      val sql_context = new org.apache.spark.sql.SQLContext(spark.sparkContext)
      val top_k_scores = articleScoresDataset.map{ all_scores =>
        val id = all_scores.id
        val top_k_sentences = find_top_K(all_scores.sentence_list,all_scores.scores,top_k)
        Article(id,top_k_sentences)
      }
      val raw_data = readContent(wordFile).select("id","title")
      val df = top_k_scores.join(raw_data,"id")
      df.toJSON.coalesce(1).write.text(outputPath)
      //      articleScoresDataset.registerTempTable(s"t${i}")
      //      var max = sql_context.sql(s"select max(scores) from t${i}")
      //      for (j<- 0 to top_k){
      //        val update_max = sql_context.sql(s"select max(scores) from t${i} where t${i}.scores<t${max}")
      //        max = update_max
      //      }
      //
      //
      //      val  = sql_context.sql(s"Select scores From t${i} t11 Where (${top_k}-1) = (Select Count(Distinct(t22.scores)) From t${i} t22 Where t22.scores > t11.scores)")
      //      sql_context.dropTempTable("t${i}")
      //      val df = ArticleWithScores.filter("scores>top_k_scores").join(data,"id")
    }
  }

  def find_top_K(array: Array[Array[String]], doubles: Array[Double],top_k: Int): Array[Array[String]] ={
    if (top_k<array.length){
      val pq = collection.mutable.PriorityQueue(0.0)
      var i=0
      for (i<-0 to doubles.length-1){
        pq.enqueue(doubles(i))
      }
      i = 0
      var top_k_score = 0.0
      for (i<-0 to top_k-1){
        top_k_score = pq.dequeue()
      }
      var res: Array[Array[String]] = Array()
      i=0
      for (i<-0 to array.length-1){
        if(doubles(i)>=top_k_score){
          res = res :+ array(i)
        }
      }
      res
    }
    else {
      array
    }
  }


  def readContent(path : String) ={
    val stringFrame = spark.read.text(path).as[String]
    val jsonFrame = spark.read.json(stringFrame)
    jsonFrame
  }
  def calculateSimilarity(first: Seq[String], second: Seq[String]): Double ={
    first.intersect(second).length / (Math.log(first.length) + Math.log(second.length))
  }

  def constructRelation(pair: Array[(Long, Array[String])]) = {
    pair.flatMap{ sent1 =>
      pair.map (sent2 => Relation(sent1._1, sent2._1,
        if (sent1._1 == sent2._1) 0 else calculateSimilarity(sent1._2, sent2._2)))
        .filter(relation => relation.weight > 0.2)
    }
  }

  def calculate_score(first: (Array[String], Int), second: (Array[String], Int), threshold: Double) = {
    val initial_score = if (first._2 != second._2) calculateSimilarity(first._1, second._1) else 0.0
    if (initial_score > threshold) initial_score else 0.0
  }

  def textRank(weightMatrix: Array[Array[Double]], neighbours : Array[Array[Int]], tolerance: Double) = {
    var oldScores = Array.fill(neighbours.size)(1.0)
    var maxDiff = 10.0
    while (maxDiff > tolerance) {
      val newScores = neighbours.zipWithIndex
        .map(ele => 0.15 * oldScores(ele._2) + 0.85 * ele._1.map(
          index => oldScores(index) * weightMatrix(index)(ele._2)).sum)
      maxDiff = newScores.zip(oldScores).map(tup => (tup._1 - tup._2).abs).max
      oldScores = newScores
    }
    oldScores
  }

  def normalizeArray(array: Array[Double]) = {
    val sum = array.sum
    array.map(ele => ele / sum)
  }
}
