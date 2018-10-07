import org.apache.spark.sql.{DataFrame, SparkSession}

object RAKE {

  def rake(inputFrame: DataFrame): Unit ={
    var stopwordFile = "src/main/resources/stopWords.txt"

    val spark = SparkSession
      .builder().master("local")
      .appName("RAKE")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    val sc = spark.sparkContext
    val stopWordText = sc.textFile(stopwordFile).collect()
    stopWordText.flatMap(_.stripMargin.split(","))
    val stopsRegex: String = stopWordText.map(w => s"\\b$w(?![\\w-])").mkString("|")
    val delimiters = "[.!?,;:\t\\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s"
    val raw = inputFrame.select("content").rdd.collect()
    for(i <- 0 until raw.length){
      val text = raw(i).toString()
      val scores = RAKE.run(text,delimiters,stopsRegex).toSeq.sortWith(_._2 > _._2)
    }

  }
  def run(text: String, delimiters: String, stopsRegex: String): Map[String, Double] = {
    val phraseList = text
      .split(delimiters)
      .flatMap(s => s.toLowerCase.split(stopsRegex).map(_.trim))
      .filterNot(w => w.isEmpty || w == " ")
    val scores = phraseList
      .map(separateWords)
      .flatMap(words => words.map((_, words.length - 1)))
      .groupBy(_._1)
      .map {
        case (word, l) =>
          val score = (l.length + l.map(_._2).sum) / (l.length * 1D)
          word -> score
      }
    phraseList
      .map(phrase => phrase -> separateWords(phrase).map(scores(_)).sum)
      .toMap
  }

  private def separateWords(phrase: String): Array[String] = {
    phrase.split("[^a-zA-Z0-9_\\+\\-/]")
      .filter(w => w.nonEmpty && !w.matches("^\\d+$"))
    //      .map(_.toLowerCase.trim)
  }
}
