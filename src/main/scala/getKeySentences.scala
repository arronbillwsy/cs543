import breeze.linalg.DenseMatrix
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object getKeySentences {

  val conf = new SparkConf().setAppName("wordcount").setMaster("local");
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {

    val i = 0;
    for( i <- 1 to 8) {
      val wordFile = s"file:///C:/Users/31476/Desktop/543/bytecup2018/bytecup.corpus.train.${i}.txt"
      val data = readContent(wordFile)
      val raw = data.select("content").rdd.collect()
      for(i <- 0 until raw.length){
        var text = raw(i).toString()
        text = text.substring(1,text.length()-1)
        val key_sentences = textRank(text)
        var sen_text = key_sentences(0)
        var m = 1
        while (m < key_sentences.length) {
          sen_text = sen_text+"."+key_sentences(i)
          i += 1
        }
        val newdf = data.withColumn("content",when(col("content")===text, sen_text).otherwise(col("content")))
        data = newdf
        print(1)
      }
    }
  }



  def readContent(path : String) ={
    val spark = SparkSession.builder().master("local").appName("readWords")
      .config("spark.some.config.option", "some-value").getOrCreate()
    import spark.implicits._
    val stringFrame = spark.read.text(path).as[String]
    val jsonFrame = spark.read.json(stringFrame)
    val data = jsonFrame.select("id","title", "content").filter("content != ''")
    data
  }

  def textRank(content: String): Array[String] ={
    val threshhold = 0
    val top_k = 10
    val separator = Array('.','!','?',';','-')
    val sentences = content.split(separator).distinct
    val i=0
    var vertices: Array[(Long,String)]=Array()
    var edges: Array[Edge[Double]] = Array()
    for (i <- 0 until sentences.length){
      val s1 = sentences(i)
      vertices :+= (i.toLong,s1)
      val j=0
      for (j <- 0 until sentences.length){
        if (i!=j){
          val s2 = sentences(j)
          val sim = cal_sen_similarity(s1,s2)
          if (sim>threshhold){
            edges :+= Edge(i.toLong,j.toLong,sim)
            edges :+= Edge(j.toLong,i.toLong,sim)
          }
        }
      }
    }
    val vRDD= sc.parallelize(vertices)
    val eRDD= sc.parallelize(edges)
    val graph = Graph(vRDD,eRDD)
    val ranks = graph.pageRank(0.0001).vertices
    val scores = vRDD.join(ranks)
    val sorted_scores = scores.sortBy(_._2._2, false)
    val key_sentences = sorted_scores.take(10).map(_._2._1).take(top_k)
    key_sentences
  }

  def cal_sen_similarity(s1 : String,s2 : String): Double ={
    val words1 = sc.parallelize(s1.split(" "))
    val words2 = sc.parallelize(s2.split(" "))
    val intersection = words1.intersection(words2)
    val sim = intersection.count()/(math.log(words1.count())*math.log(words2.count()))
    sim
  }

}
