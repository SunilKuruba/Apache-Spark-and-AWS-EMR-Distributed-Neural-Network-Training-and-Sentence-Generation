import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster

import java.io.{BufferedWriter, File, FileWriter, IOException}

object Main {
  private val logger = LoggerFactory.getLogger(this.getClass)

  /**
   * Enumeration representing the different environments in which the job can run.
   * Available environments are:
   * - `prod`: Production environment.
   * - `local`: Local development environment.
   * - `test`: Testing environment.
   */
  sealed trait Environment
  private object Environment {
    case object prod extends Environment
    case object local extends Environment
    case object test extends Environment
  }

  /** The environment in which the job is currently running. Defaults to `local`. */
  var environment: Environment = Environment.local

  def main(args: Array[String]): Unit = {
    // Load configuration settings
    val config = ConfigFactory.load()
    val sparkContext = new SparkContext(getSparkConf)
    val numOfOutputPredictions = config.getInt("model.numOfOutputPredictions")

    // Determine input and output file paths based on command line arguments or configuration
    val inputFilePath = if (args.length > 0) args(0) else config.getString(s"io.inputdir.$environment")
    val outputResultFilePath = if (args.length > 1) args(1) else config.getString(s"io.outputResult.$environment")
    val outputStatsFilePath = if (args.length > 2) args(2) else config.getString(s"io.outputStats.$environment")

    // Initialize the metrics writer for logging training metrics
    val metricsWriter = new BufferedWriter(new FileWriter(outputStatsFilePath))
    try {
      metricsWriter.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed," +
        "\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines,\tMemoryUsed\n")

      // Read input text and preprocess
      val textRDD = sparkContext.textFile(inputFilePath)
        .map(_.trim)
        .filter(_.nonEmpty)
        .cache()

      // Log initial statistics
      logger.info(s"Number of partitions: ${textRDD.getNumPartitions}")
      logger.info(s"Total number of lines: ${textRDD.count()}")

      // Train the model
      val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
        .batchSizePerWorker(32)
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)
        .build()

      val trainedModel = new Train().train(sparkContext, textRDD, metricsWriter, 1, trainingMaster)
      logger.info("Model training completed.")

      // Generate sample text using the trained model
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)
      logger.info("Tokenizer fitted on the text data.")

      val seedToken = "The little cat"
      val generatedText = new TextOutput().generateText(trainedModel, tokenizer, seedToken, numOfOutputPredictions)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      writeToFile(s"Generated text: $seedToken -> $cleanedText", outputResultFilePath)

    } catch {
      case e: IOException =>
        logger.error("Error occurred while writing metrics or reading input data.", e)
    } finally {
      metricsWriter.close()
      sparkContext.stop()
      logger.info("SparkContext stopped successfully.")
    }
  }

  private def getSparkConf: SparkConf = {
    new SparkConf()
      .setAppName("Sunil's Spark LLM")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(
        classOf[MultiLayerNetwork],
        classOf[INDArray],
        classOf[Array[Byte]],
        classOf[org.nd4j.linalg.api.buffer.DataBuffer]
      ))
  }

  /**
   * Utility method to write log messages to a file.
   * @param message The message to write to the file.
   * @param filePath The path to the file where the message will be written.
   */
  private def writeToFile(message: String, filePath: String): Unit = {
    val file = new File(filePath)
    val writer = new BufferedWriter(new FileWriter(file))
    try {
      writer.write(message)
    } catch {
      case e: IOException =>
        logger.error("Error occurred while writing the log message to the file.", e)
    } finally {
      writer.close()
    }
  }
}
