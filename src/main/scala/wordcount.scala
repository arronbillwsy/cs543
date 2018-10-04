import org.apache.spark.{SparkConf, SparkContext}

object wordcount {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("wordcount").setMaster("local");
    val sc = new SparkContext(conf)
    val wordFile = "file:///C:/Users/31476/Desktop/543/bytecup2018/bytecup.corpus.train.0.txt"

    val input = sc.textFile(wordFile, 2).cache()
    val lines = input.flatMap(line=>line.split(" "))
    val count = lines.map(word => (word,1)).reduceByKey{case (x,y)=>x+y}
    val output = count.saveAsTextFile("/C:/Users/31476/Desktop/543/bytecup2018/res.txt")
  }
}
