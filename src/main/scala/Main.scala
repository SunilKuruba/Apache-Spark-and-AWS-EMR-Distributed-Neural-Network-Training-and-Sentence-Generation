import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

import java.io.{BufferedWriter, FileWriter}

object Main {
  /**
   * Enumeration representing the different environments in which the job can run.
   * Available environments are:
   * - `prod`: Production environment.
   * - `local`: Local development environment.
   * - `test`: Testing environment.
   */
  sealed trait Environment
  object Environment {
    case object prod extends Environment
    case object local extends Environment
    case object test extends Environment
  }

  /** The environment in which the job is currently running. Defaults to `local`. */
  var environment: Environment = Environment.local

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.load
    val sparkContext = new SparkContext(getSparkConf)
    val inputFilePath = config.getString("io.inputdir."+environment)
    val outputFilePath = config.getString("io.outputdir."+environment)
    val metricsWriter = new BufferedWriter(new FileWriter(outputFilePath))
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
