import org.apache.spark.sql.SparkSession


object WordCountSpark {
  def main(args: Array[String]): Unit = {
    // Create Spark session with security disabled
    val spark = SparkSession.builder()
      .appName("WordCount")
      .master("local[*]")  // Use as many worker threads as logical cores
      .getOrCreate()

    val distFile = "/Users/sunilkuruba/Desktop/tiny_input.txt"

    // Read the text file
    val textFile = spark.sparkContext.textFile(distFile)
//
//    val tokenizedData = partitionedData.flatMap(line => line.split(" "))

//    // Perform word count
    val counts = textFile
      .flatMap(line => line.split("\\W+")) // Split lines into words
      .filter(_.nonEmpty)
      .map(word => (word, 1))            // Map each word to a tuple
      .reduceByKey(_ + _)                // Reduce by key to count occurrences


    // Save the counts to a text file
    counts.saveAsTextFile("src/main/resources/output/WordCountSpark-5") // Specify an output directory

    // Stop the Spark session
    spark.stop()
  }
}
