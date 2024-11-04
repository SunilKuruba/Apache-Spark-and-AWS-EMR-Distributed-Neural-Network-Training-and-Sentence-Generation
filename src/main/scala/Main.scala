import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory

import java.io.IOException
import java.time.Instant

object Main extends Serializable{
  private val logger = LoggerFactory.getLogger(this.getClass)

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
  var environment: Environment = Environment.prod

  def main(args: Array[String]): Unit = {
    // Load configuration settings
    val config = ConfigFactory.load()
    val sparkContext = new SparkContext(getSparkConf)
    val numOfOutputPredictions = config.getInt("model.numOfOutputPredictions")
    val numOfEpoch = config.getInt("model.numOfEpoch")
    val instant = Instant.now().toString

    val seedToken  =  if (args.length > 0) args(0) else "The little cat"

    // Determine input and output file paths based on command line arguments or configuration
    val inputFilePath = if (args.length > 1) args(1) else config.getString(s"io.inputdir.$environment")
    val outputResultFilePath = if (args.length > 1) args(2) else config.getString(s"io.outputResult.$environment")
    val outputStatsFilePath = if (args.length > 1) args(3) else config.getString(s"io.outputStats.$environment")

    val inputData: String = if (inputFilePath.startsWith("s3")) {
      environment = Environment.prod
      val s3InputFileSystem = new S3FileSystem(sparkContext, inputFilePath)
      s3InputFileSystem.read()
    } else {
      val localInputFileSystem = new LocalFileSystem(inputFilePath)
      localInputFileSystem.read()
    }

    val resultFile: FileSystem = if (inputFilePath.startsWith("s3")) {
      val s3ResultFile = new S3FileSystem(sparkContext, outputResultFilePath.replace("{instant}", instant))
      s3ResultFile.create()
      s3ResultFile
    } else {
      val localResultFile = new LocalFileSystem(outputResultFilePath.replace("{instant}", instant))
      localResultFile.create()
      localResultFile
    }

    val statsFile: FileSystem = if (inputFilePath.startsWith("s3")) {
      val s3StatsFile = new S3FileSystem(sparkContext, outputStatsFilePath.replace("{instant}", instant))
      s3StatsFile
    } else {
      val localStatsFile = new LocalFileSystem(outputStatsFilePath.replace("{instant}", instant))
      localStatsFile.create()
      localStatsFile
    }

    // Initialize the metrics writer for logging training metrics
    try {
      statsFile.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed," +
        "\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines,\tMemoryUsed\n")

      // Read input text and preprocess
      val textRDD = sparkContext.parallelize(inputData.split("\n"))

      // Log initial statistics
      logger.info(s"Number of partitions: ${textRDD.getNumPartitions}")
      logger.info(s"Total number of lines: ${textRDD.count()}")

      // Train the model
      val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
        .batchSizePerWorker(32)
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)
        .build()

      val trainedModel = new Train().train(sparkContext, textRDD, statsFile, numOfEpoch, trainingMaster)
      logger.info("Model training completed.")

      // Generate sample text using the trained model
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)
      logger.info("Tokenizer fitted on the text data.")

      val generatedText = new TextOutput().generateText(trainedModel, tokenizer, seedToken, numOfOutputPredictions)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      resultFile.write(s"Generated text: $seedToken -> $cleanedText")
      logger.info(s"Results are found in /output/$instant")

    } catch {
      case e: IOException =>
        logger.error("Error occurred while writing metrics or reading input data.", e)
    } finally {
      statsFile.close()
      resultFile.close()
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
}
