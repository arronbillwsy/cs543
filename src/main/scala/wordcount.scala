import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object wordcount {

  def main(args: Array[String]): Unit = {
    val i = 0
    var filepath = s"/C:/Users/31476/Desktop/543/bytecup2018/processed_train.${i}.txt"
    val data = readWords(filepath)
    val raw = data.select("words").rdd.collect()
    for(j <- 0 until raw.length){
      val text = raw(j).toString()

    }



    val conf = new SparkConf().setAppName("wordcount").setMaster("local");
    val sc = new SparkContext(conf)







    for( i <- 1 to 8){
      val wordFile = s"file:///C:/Users/31476/Desktop/543/bytecup2018/bytecup.corpus.train.${i}.txt"
      val input = sc.textFile(wordFile, 2).cache()
      val lines = input.flatMap(line=>line.split(" "))
//      count = lines.map(word => (word,1)).reduceByKey{case (x,y)=>x+y}.union(count)
      println(i)
    }
//    count.reduceByKey{case (x,y)=>x+y}
//    val output = count.saveAsTextFile("/C:/Users/31476/Desktop/543/bytecup2018/res")
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
