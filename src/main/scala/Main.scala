import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

import java.io.{BufferedWriter, FileWriter}

object Main {
  def main(args: Array[String]): Unit = {
    val sparkContext = new SparkContext(getSparkConf)

    //  val metricsFilePath = "s3://hw2-spark-llm/output/"
    val metricsFilePath = "src/main/resources/training_metrics.csv"
    val inputFilePath = "src/main/resources/tiny_input.txt"

    val metricsWriter = new BufferedWriter(new FileWriter(metricsFilePath))
    metricsWriter.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines, \tMemoryUsed\n")

    try {
      val textRDD = sparkContext.textFile(inputFilePath)
        .map(_.trim)
        .filter(_.nonEmpty)
        .cache()

      // Print initial statistics
      println(s"Number of partitions: ${textRDD.getNumPartitions}")
      println(s"Total number of lines: ${textRDD.count()}")

      val trainedModel = new Train().train(sparkContext, textRDD, metricsWriter, 1)

      // Generate sample text
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)

      val prefix = "ocean"
      val generatedText = new TextOutput().generateText(trainedModel, tokenizer, prefix, 100)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      println(s"Cleaned text: $prefix -> $cleanedText")

    } finally {
      metricsWriter.close()
      sparkContext.stop()
    }
  }

  def getSparkConf: SparkConf = {
    new SparkConf()
      .setAppName("Sunil's Spark LLM")
      .setMaster("local[*]")
      .set("spark.executor.memory", "4g")
      .set("spark.driver.memory", "4g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryoserializer.buffer.max", "512m")
      .set("spark.kryoserializer.buffer", "256m")
      .registerKryoClasses(Array(
        classOf[MultiLayerNetwork],
        classOf[INDArray],
        classOf[Array[Byte]],
        classOf[org.nd4j.linalg.api.buffer.DataBuffer]
      ))
  }
}
