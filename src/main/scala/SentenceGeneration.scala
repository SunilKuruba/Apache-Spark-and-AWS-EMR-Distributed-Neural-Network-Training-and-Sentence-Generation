import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.ops.transforms.Transforms
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.schedule.{ExponentialSchedule, ScheduleType}

import java.io.{ByteArrayInputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}

class SentenceGenerationImpl extends Serializable {
  val vocabularySize: Int = 3000
  val embeddingSize: Int = 32
  val windowSize: Int = 1
  val batchSize: Int = 16

  class SimpleTokenizer extends Serializable {
    private var wordToIndex = Map[String, Int]()
    private var indexToWord = Map[Int, String]()
    private var currentIdx = 0

    def fit(texts: Seq[String]): Unit = {
      texts.flatMap(_.split("\\s+")).distinct.foreach { word =>
        if (!wordToIndex.contains(word)) {
          wordToIndex += (word -> currentIdx)
          indexToWord += (currentIdx -> word)
          currentIdx += 1
        }
      }
    }

    def encode(text: String): Seq[Int] = {
      text.split("\\s+").map(word => wordToIndex.getOrElse(word, 0))
    }

    def decode(indices: Seq[Int]): String = {
      indices.map(idx => indexToWord.getOrElse(idx, "")).mkString(" ")
    }
  }

  private def serializeModel(model: MultiLayerNetwork): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    try {
      val oos = new ObjectOutputStream(baos)
      oos.writeObject(model.params())
      oos.writeObject(model.getLayerWiseConfigurations)
      oos.close()
      baos.toByteArray
    } finally {
      baos.close()
    }
  }

  private def deserializeModel(bytes: Array[Byte]): MultiLayerNetwork = {
    val bais = new ByteArrayInputStream(bytes)
    try {
      val ois = new ObjectInputStream(bais)
      val params = ois.readObject().asInstanceOf[INDArray]
      val conf = ois.readObject().asInstanceOf[org.deeplearning4j.nn.conf.MultiLayerConfiguration]
      val model = new MultiLayerNetwork(conf)
      model.init()
      model.setParams(params)
      model
    } finally {
      bais.close()
    }
  }

  // Create sliding windows for training data
  def createSlidingWindows(tokens: Seq[Int]): Seq[(Seq[Int], Int)] = {
    tokens.sliding(windowSize + 1).map { window =>
      (window.init, window.last)
    }.toSeq
  }

  // Convert sequence to embedding matrix with positional encoding
  def createEmbeddingMatrix(sequence: Seq[Int]): INDArray = {
    val embedding = Nd4j.zeros(1, embeddingSize, sequence.length)

    // Create word embeddings
    sequence.zipWithIndex.foreach { case (token, pos) =>
      val tokenEmbedding = Nd4j.randn(1, embeddingSize).mul(0.1)
      embedding.putSlice(pos, tokenEmbedding)
    }

    // Add positional encodings
    for (pos <- sequence.indices) {
      for (i <- 0 until embeddingSize) {
        val angle = pos / math.pow(10000, (2 * i).toFloat / embeddingSize)
        if (i % 2 == 0) {
          embedding.putScalar(Array(0, i, pos), embedding.getDouble(0, i, pos) + math.sin(angle))
        } else {
          embedding.putScalar(Array(0, i, pos), embedding.getDouble(0, i, pos) + math.cos(angle))
        }
      }
    }

    embedding
  }

  def selfAttention(input: INDArray): INDArray = {
    val Array(batchSize, sequenceLength, embedSize) = input.shape()

    // Create query, key, and value matrices for each batch independently
    val query = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val key = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val value = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)

    // Ensure query, key, and value are initialized properly
    if (query.isEmpty || key.isEmpty || value.isEmpty) {
      return Nd4j.empty()
    }

    // Compute the dot product between queries and keys
    val scores = query
      .tensorAlongDimension(0, 1, 2)
      .mmul(key.tensorAlongDimension(0, 1, 2).transpose())
      .div(math.sqrt(embedSize))

    // Apply softmax along the last dimension to get attention weights
    val attentionWeights = Transforms.softmax(scores)

    // Multiply the weights with the values
    val attendedOutput = attentionWeights
      .tensorAlongDimension(0, 1, 2)
      .mmul(value.tensorAlongDimension(0, 1, 2))

    attendedOutput.reshape(batchSize, sequenceLength, embedSize)
  }

  case class TrainingMetrics(
                              epoch: Int,
                              batchTime: Long,
                              loss: Double,
                              accuracy: Double,
                              memoryUsed: Long,
                              batchesProcessed: Long
                            )

  // Custom listener for collecting training metrics
  class CustomTrainingListener extends ScoreIterationListener(10) {
    private var currentScore: Double = 0.0

    override def iterationDone(model: org.deeplearning4j.nn.api.Model, iteration: Int, epochNum: Int): Unit = {
      super.iterationDone(model, iteration, epochNum)
      currentScore = model.score()
    }

    def getLastScore: Double = currentScore
  }

  def buildModel(validationIterator : DataSetIterator): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 0.005, 0.9)))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(embeddingSize * windowSize)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(512)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(2, new OutputLayer.Builder()
        .nIn(128)
        .nOut(vocabularySize)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    // Add custom listener for monitoring
    val listener = new CustomTrainingListener
    model.setListeners(listener, new GradientNormListener(10), new EvaluativeListener(validationIterator, 1))

    model
  }

  def createValidationDataSetIterator(validationDataRDD: RDD[String], tokenizer: SimpleTokenizer): DataSetIterator = {
    // Process the validation data to create features and labels
    val validationData = validationDataRDD.flatMap { text =>
      val tokens = tokenizer.encode(text)
      createSlidingWindows(tokens).map { case (inputSeq, label) =>
        val inputArray = Nd4j.zeros(1, embeddingSize * windowSize)
        val labelArray = Nd4j.zeros(1, vocabularySize)

        // Convert input sequence and label to ND4J arrays
        val embedding = createEmbeddingMatrix(inputSeq)
        val attentionOutput = selfAttention(embedding)
        if (!attentionOutput.isEmpty) {
          val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
          inputArray.putRow(0, flattenedAttention)
          labelArray.putScalar(Array(0, label), 1.0)

          new DataSet(inputArray, labelArray)
        }
        new DataSet()
      }
    }.collect().toList.asJava

    // Create a ListDataSetIterator with a batch size of 1 (or adjust as needed)
    new ListDataSetIterator(validationData, batchSize)
  }

  // Modify the train method to use serialization
  def train(sc: SparkContext, textRDD: RDD[String], metricsWriter: BufferedWriter, epochs: Int): MultiLayerNetwork = {
    val tokenizer = new SimpleTokenizer()
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    // Split textRDD into training and validation sets
    val Array(trainingDataRDD, validationDataRDD) = textRDD.randomSplit(Array(0.8, 0.2))

    // Generate validation DataSetIterator
    val validationDataSetIterator = createValidationDataSetIterator(validationDataRDD, tokenizer)

    val model = buildModel(validationDataSetIterator)
    var currentModelBytes = serializeModel(model)
    var broadcastModel = sc.broadcast(currentModelBytes)

    val batchProcessedAcc = sc.longAccumulator("batchesProcessed")
    val totalLossAcc = sc.doubleAccumulator("totalLoss")
    val correctPredictionsAcc = sc.longAccumulator("correctPredictions")
    val totalPredictionsAcc = sc.longAccumulator("totalPredictions")

    for (epoch <- 1 to epochs) {
      val epochStartTime = System.currentTimeMillis()
      println(s"Starting epoch $epoch")

      // Retrieve and print the learning rate from the optimizer (Adam)
      val learningRate = model.getLayerWiseConfigurations.getConf(0).getLayer
        .asInstanceOf[org.deeplearning4j.nn.conf.layers.BaseLayer]
        .getIUpdater.asInstanceOf[Adam].getLearningRate(epoch, epochs) // Pass the current epoch to get effective rate
      println(s"Effective learning rate for epoch $epoch: $learningRate")

      batchProcessedAcc.reset()
      totalLossAcc.reset()
      correctPredictionsAcc.reset()
      totalPredictionsAcc.reset()

      val samplesRDD = trainingDataRDD.flatMap { text =>
        val tokens = broadcastTokenizer.value.encode(text)
        createSlidingWindows(tokens)
      }.persist()

      val processedRDD = samplesRDD.mapPartitions { partition =>
        val localModel = deserializeModel(broadcastModel.value)
        val batchBuffer = new scala.collection.mutable.ArrayBuffer[(Seq[Int], Int)]()
        var localLoss = 0.0
        var localCorrect = 0L
        var localTotal = 0L

        partition.foreach { sample =>
          batchBuffer += sample
          if (batchBuffer.size >= batchSize) {
            val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
            localLoss += loss
            localCorrect += correct
            localTotal += total
            batchBuffer.clear()
            batchProcessedAcc.add(1)
          }
        }

        if (batchBuffer.nonEmpty) {
          val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
          localLoss += loss
          localCorrect += correct
          localTotal += total
          batchProcessedAcc.add(1)
        }

        totalLossAcc.add(localLoss)
        correctPredictionsAcc.add(localCorrect)
        totalPredictionsAcc.add(localTotal)

        Iterator.single(serializeModel(localModel))
      }

      val updatedModels = processedRDD.collect()
      if (updatedModels.nonEmpty) {
        val averagedModel = if (updatedModels.length > 1) {
          val models = updatedModels.map(deserializeModel)
          averageModels(models)
        } else {
          deserializeModel(updatedModels(0))
        }

        broadcastModel.destroy()
        currentModelBytes = serializeModel(averagedModel)
        broadcastModel = sc.broadcast(currentModelBytes)

        val epochDuration = System.currentTimeMillis() - epochStartTime
        val avgLoss = totalLossAcc.value / batchProcessedAcc.value
        val accuracy = if (totalPredictionsAcc.value > 0) {
          correctPredictionsAcc.value.toDouble / totalPredictionsAcc.value
        } else 0.0

        println(f"""
                   |Epoch $epoch Statistics:
                   |Duration: ${epochDuration}ms
                   |Average Loss: $avgLoss%.4f
                   |Accuracy: ${accuracy * 100}%.2f%%
                   |Batches Processed: ${batchProcessedAcc.value}
                   |Predictions Made: ${totalPredictionsAcc.value}
                   |Memory Used: ${Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()}B
      """.stripMargin)
        // Differentiating between executor memory and driver memory.
        val executorMemoryStatus = sc.getExecutorMemoryStatus.map { case (executor, (maxMemory, remainingMemory)) =>
          s"$executor: Max Memory = $maxMemory, Remaining Memory = $remainingMemory"
        }
        println(s"Executor Memory Status:\n${executorMemoryStatus.mkString("\n")}")
        // Write metrics to the CSV file
        metricsWriter.write(f"$epoch,$learningRate%.6f,$avgLoss%.4f,${accuracy * 100}%.2f,${batchProcessedAcc.value},${totalPredictionsAcc.value},$epochDuration,${textRDD.getNumPartitions},${textRDD.count()},,${executorMemoryStatus.mkString("\n")}\n")
      }

      samplesRDD.unpersist()
      val epochEndTime = System.currentTimeMillis()
      println(s"Time per Epoch: ${epochEndTime - epochStartTime} ms")
    }
    // Close the writer after all epochs are done
    metricsWriter.close()
    deserializeModel(broadcastModel.value)
  }



  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): (Double, Long, Long) = {
    val inputArray = Nd4j.zeros(batch.size, embeddingSize * windowSize)
    val labelsArray = Nd4j.zeros(batch.size, vocabularySize)

    batch.zipWithIndex.foreach { case ((sequence, label), idx) =>
      val embedding = createEmbeddingMatrix(sequence)
      val attentionOutput = selfAttention(embedding)
      if (!attentionOutput.isEmpty) {
        val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
        inputArray.putRow(idx, flattenedAttention)
        labelsArray.putScalar(Array(idx, label), 1.0)
      }
    }

    // Train on batch
    model.fit(inputArray, labelsArray)

    //    val learningRate = model.getConfig..ra.getLearningRate
    //    println(s"Current Learning Rate: $learningRate")


    // Calculate metrics
    val output = model.output(inputArray)
    val predictions = Nd4j.argMax(output, 1)
    val labels = Nd4j.argMax(labelsArray, 1)
    // Calculate the number of correct predictions
    val correct = predictions.eq(labels).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
      .sumNumber().longValue()

    (model.score(), correct, batch.size)
  }

  private def averageModels(models: Array[MultiLayerNetwork]): MultiLayerNetwork = {
    val firstModel = models(0)
    if (models.length == 1) return firstModel

    val params = models.map(_.params())
    val avgParams = params.reduce((a, b) => a.add(b)).div(models.length)

    val result = new MultiLayerNetwork(firstModel.getLayerWiseConfigurations)
    result.init()
    result.setParams(avgParams)
    result
  }

  /// Modified text generation with temperature sampling
  def generateText(model: MultiLayerNetwork, tokenizer: SimpleTokenizer, seedText: String, length: Int, temperature: Double = 0.7): String = {
    var currentSequence = tokenizer.encode(seedText).takeRight(windowSize)
    val generated = new ArrayBuffer[Int]()
    val rand = new Random()

    def sampleWithTemperature(logits: INDArray, temp: Double): Int = {
      val scaled = logits.div(temp)
      val expScaled = Transforms.exp(scaled)
      val probs = expScaled.div(expScaled.sum(1))

      // Convert to probabilities and sample
      val probArray = Array.ofDim[Double](probs.columns())
      for (i <- 0 until probs.columns()) {
        probArray(i) = probs.getDouble(Long.box(i))
      }

      // Sample using cumulative probabilities
      val cumSum = probArray.scanLeft(0.0)(_ + _).tail
      val sample = rand.nextDouble()
      cumSum.zipWithIndex.find(_._1 >= sample).map(_._2).getOrElse(0)
    }

    for (_ <- 1 to length) {
      val embedding = createEmbeddingMatrix(currentSequence)
      val attentionOutput = selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)

      val output = model.output(flattenedAttention)
      val nextTokenIndex = sampleWithTemperature(output, temperature)

      generated += nextTokenIndex
      currentSequence = (currentSequence.tail :+ nextTokenIndex).takeRight(windowSize)
    }

    tokenizer.decode(generated)
  }

  def getSparkConf(): SparkConf = {
    new SparkConf()
      .setAppName("DistributedLanguageModel")
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

  class GradientNormListener(logFrequency: Int) extends IterationListener {
    private var iteration = 0

    //  override def invoked(): Boolean = true

    //  override def invoke(): Unit = {}

    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      if (iteration % logFrequency == 0) {
        // Get the gradients
        val gradients: INDArray = model.gradient().gradient()

        val gradientMean = gradients.meanNumber().doubleValue()
        val gradientMax = gradients.maxNumber().doubleValue()
        val gradientMin = gradients.minNumber().doubleValue()
        println(s"Iteration $iteration: Gradient Mean = $gradientMean, Max = $gradientMax, Min = $gradientMin")
      }
    }
  }

}

object SimpleLanguageModel {
  def main(args: Array[String]): Unit = {
    // Configure Spark with appropriate resources
    val model = new SentenceGenerationImpl()
    val sc = new SparkContext(model.getSparkConf())

    val metricsFilePath = "src/main/resources/training_metrics.csv"

    val metricsWriter = new BufferedWriter(new FileWriter(metricsFilePath))
    metricsWriter.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines, \tMemoryUsed\n")

    try {
      // Enable logging
      sc.setLogLevel("INFO")

      val filePath = "src/main/resources/tiny_input.txt"
      val textRDD = sc.textFile(filePath)
        .map(_.trim)
        .filter(_.nonEmpty)
        .cache()

      // Print initial statistics
      println(s"Number of partitions: ${textRDD.getNumPartitions}")
      println(s"Total number of lines: ${textRDD.count()}")

      val trainedModel = model.train(sc, textRDD, metricsWriter, 1)

      // Generate sample text
      val tokenizer = new model.SimpleTokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)

      val prefix = "scientist"
      val generatedText = model.generateText(trainedModel, tokenizer, prefix, 100)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      println(s"Cleaned text: $prefix -> $cleanedText")

    } finally {
      metricsWriter.close()
      sc.stop()
    }
  }
}