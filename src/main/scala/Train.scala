import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.LoggerFactory

import java.io.BufferedWriter

/**
 * The Train class handles the training of a neural network using Spark.
 * It uses a self-attention mechanism for embeddings and handles distributed training.
 */
class Train extends Serializable {
  private val logger = LoggerFactory.getLogger(classOf[Train])
  private val config = ConfigFactory.load()

  // Configuration parameters from the config file
  private val vocabularySize: Int = config.getInt("model.vocabularySize")
  private val embeddingSize: Int = config.getInt("model.embeddingSize")
  private val windowSize: Int = config.getInt("model.windowSize")
  private val batchSize: Int = config.getInt("model.batchSize")
  private val modelClass = new NNModel()

  /**
   * Applies a self-attention mechanism to the input tensor.
   */
  def selfAttention(input: INDArray): INDArray = {
    logger.debug("Applying self-attention to input tensor.")
    val Array(batchSize, sequenceLength, embedSize) = input.shape()

    val query = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val key = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val value = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)

    val scores = query
      .tensorAlongDimension(0, 1, 2)
      .mmul(key.tensorAlongDimension(0, 1, 2).transpose())
      .div(math.sqrt(embedSize))

    val attentionWeights = Transforms.softmax(scores)
    logger.debug("Calculated attention weights.")

    val attendedOutput = attentionWeights
      .tensorAlongDimension(0, 1, 2)
      .mmul(value.tensorAlongDimension(0, 1, 2))

    logger.debug("Generated attended output tensor.")
    attendedOutput.reshape(batchSize, sequenceLength, embedSize)
  }

  /**
   * Trains the neural network model using the provided RDD and configuration.
   */
  def train(sc: SparkContext, textRDD: RDD[String], metricsWriter: BufferedWriter, epochs: Int): MultiLayerNetwork = {
    logger.info("Starting training process.")
    val tokenizer = new Tokenizer()
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    val Array(trainingDataRDD, validationDataRDD) = textRDD.randomSplit(Array(0.8, 0.2))
    logger.info("Split data into training and validation sets.")

    val validationDataSetIterator = createValidationDataSetIterator(validationDataRDD, tokenizer)
    val initialModel = modelClass.buildModel(validationDataSetIterator)

    val finalModel = (1 to epochs).foldLeft(initialModel) { (currentModel, epoch) =>
      trainEpoch(sc, trainingDataRDD, currentModel, broadcastTokenizer, epoch, epochs, metricsWriter)
    }

    metricsWriter.close()
    finalModel
  }

  /**
   * Trains a single epoch and returns the updated model.
   */
  private def trainEpoch(
                          sc: SparkContext,
                          trainingDataRDD: RDD[String],
                          model: MultiLayerNetwork,
                          broadcastTokenizer: org.apache.spark.broadcast.Broadcast[Tokenizer],
                          epoch: Int,
                          totalEpochs: Int,
                          metricsWriter: BufferedWriter
                        ): MultiLayerNetwork = {
    val epochStartTime = System.currentTimeMillis()
    logger.info(s"Starting epoch $epoch.")

    val learningRate = model.getLayerWiseConfigurations.getConf(0).getLayer
      .asInstanceOf[org.deeplearning4j.nn.conf.layers.BaseLayer]
      .getIUpdater.asInstanceOf[Adam].getLearningRate(epoch, totalEpochs)
    logger.info(s"Effective learning rate for epoch $epoch: $learningRate")

    // Accumulators for metrics
    val batchProcessedAcc = sc.longAccumulator("batchesProcessed")
    val totalLossAcc = sc.doubleAccumulator("totalLoss")
    val correctPredictionsAcc = sc.longAccumulator("correctPredictions")
    val totalPredictionsAcc = sc.longAccumulator("totalPredictions")

    val broadcastModel = sc.broadcast(modelClass.serializeModel(model))

    val samplesRDD = trainingDataRDD
      .flatMap(text => modelClass.createSlidingWindows(broadcastTokenizer.value.encode(text)))
      .persist()

    val processedRDD = samplesRDD.mapPartitions { partition =>
      val localModel = modelClass.deserializeModel(broadcastModel.value)

      partition
        .grouped(batchSize)
        .map { batch =>
          val (loss, correct, total) = processBatch(localModel, batch)
          totalLossAcc.add(loss)
          correctPredictionsAcc.add(correct)
          totalPredictionsAcc.add(total)
          batchProcessedAcc.add(1)
          modelClass.serializeModel(localModel)
        }
    }

    val updatedModels = processedRDD.collect()
    val newModel = if (updatedModels.nonEmpty) {
      val averagedModel = if (updatedModels.length > 1) {
        averageModels(updatedModels.map(modelClass.deserializeModel))
      } else {
        modelClass.deserializeModel(updatedModels.head)
      }

      logEpochMetrics(
        epoch,
        epochStartTime,
        learningRate,
        totalLossAcc.value,
        batchProcessedAcc.value,
        correctPredictionsAcc.value,
        totalPredictionsAcc.value,
        metricsWriter
      )

      averagedModel
    } else {
      model
    }

    broadcastModel.destroy()
    samplesRDD.unpersist()
    newModel
  }

  /**
   * Processes a batch of training data and updates the model.
   */
  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): (Double, Long, Long) = {
    logger.debug("Processing a batch of data.")
    val (inputArray, labelsArray) = createBatchArrays(batch)
    val batchDataSet = new DataSet(inputArray, labelsArray)

    val loss = model.score(batchDataSet)
    model.fit(batchDataSet)

    val predictions = model.predict(inputArray).toSeq
    val correctPredictions = predictions.zipWithIndex.count { case (pred, idx) => pred == batch(idx)._2 }

    (loss, correctPredictions, predictions.length)
  }

  /**
   * Creates input and label arrays for a batch.
   */
  private def createBatchArrays(batch: Seq[(Seq[Int], Int)]): (INDArray, INDArray) = {
    val inputArray = Nd4j.zeros(batch.size, embeddingSize * windowSize)
    val labelsArray = Nd4j.zeros(batch.size, vocabularySize)

    batch.zipWithIndex.foreach { case ((sequence, label), idx) =>
      val embedding = modelClass.createEmbeddingMatrix(sequence)
      val attentionOutput = selfAttention(embedding)
      if (!attentionOutput.isEmpty) {
        val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
        inputArray.putRow(idx, flattenedAttention)
        labelsArray.putScalar(Array(idx, label), 1.0)
      }
    }

    (inputArray, labelsArray)
  }

  /**
   * Averages the parameters of multiple models.
   */
  private def averageModels(models: Seq[MultiLayerNetwork]): MultiLayerNetwork = {
    logger.debug("Averaging parameters of multiple models.")
    val masterModel = models.head
    val avgParams = models
      .map(_.params())
      .reduce((a, b) => a.add(b))
      .div(models.length)

    val result = new MultiLayerNetwork(masterModel.getLayerWiseConfigurations)
    result.init()
    result.setParams(avgParams)
    result
  }

  /**
   * Logs metrics for an epoch.
   */
  private def logEpochMetrics(
                               epoch: Int,
                               epochStartTime: Long,
                               learningRate: Double,
                               totalLoss: Double,
                               batchesProcessed: Long,
                               correctPredictions: Long,
                               totalPredictions: Long,
                               metricsWriter: BufferedWriter
                             ): Unit = {
    val epochDuration = System.currentTimeMillis() - epochStartTime
    val avgLoss = totalLoss / batchesProcessed
    val accuracy = if (totalPredictions > 0) {
      correctPredictions.toDouble / totalPredictions
    } else 0.0

    logger.info(
      f"Epoch $epoch statistics: Duration: ${epochDuration}ms, " +
        f"Average Loss: $avgLoss%.4f, " +
        f"Accuracy: ${accuracy * 100}%.2f%%, " +
        f"Batches Processed: $batchesProcessed, " +
        f"Predictions Made: $totalPredictions"
    )

    metricsWriter.write(
      f"$epoch,$learningRate%.6f,$avgLoss%.4f,${accuracy * 100}%.2f,$batchesProcessed,$totalPredictions,$epochDuration\n"
    )
  }

  /**
   * Creates a validation DataSetIterator.
   */
  private def createValidationDataSetIterator(
                                               validationDataRDD: RDD[String],
                                               tokenizer: Tokenizer
                                             ): DataSetIterator = {
    logger.debug("Creating validation DataSetIterator.")
    val samples = validationDataRDD
      .flatMap(text => modelClass.createSlidingWindows(tokenizer.encode(text)))
      .collect()

    val (inputArray, labelsArray) = createBatchArrays(samples)
    val validationDataSet = new DataSet(inputArray, labelsArray)
    new ListDataSetIterator(validationDataSet.asList(), batchSize)
  }
}