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

import java.io.BufferedWriter
import scala.jdk.CollectionConverters.seqAsJavaListConverter

class Train extends Serializable{
  val vocabularySize: Int = 3000
  val embeddingSize: Int = 32
  val windowSize: Int = 1
  val batchSize: Int = 16
  private val modelClass = new NNModel()

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

  // Modify the train method to use serialization
  def train(sc: SparkContext, textRDD: RDD[String], metricsWriter: BufferedWriter, epochs: Int): MultiLayerNetwork = {
    val tokenizer = new Tokenizer()
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    // Split textRDD into training and validation sets
    val Array(trainingDataRDD, validationDataRDD) = textRDD.randomSplit(Array(0.8, 0.2))

    // Generate validation DataSetIterator
    val validationDataSetIterator = createValidationDataSetIterator(validationDataRDD, tokenizer)

    val model = modelClass.buildModel(validationDataSetIterator)
    var currentModelBytes = modelClass.serializeModel(model)
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
        modelClass.createSlidingWindows(tokens)
      }.persist()

      val processedRDD = samplesRDD.mapPartitions { partition =>
        val localModel = modelClass. deserializeModel(broadcastModel.value)
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

        Iterator.single(modelClass.serializeModel(localModel))
      }

      val updatedModels = processedRDD.collect()
      if (updatedModels.nonEmpty) {
        val averagedModel = if (updatedModels.length > 1) {
          val models = updatedModels.map(modelClass.deserializeModel)
          averageModels(models)
        } else {
          modelClass.deserializeModel(updatedModels(0))
        }

        broadcastModel.destroy()
        currentModelBytes = modelClass.serializeModel(averagedModel)
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
    modelClass.deserializeModel(broadcastModel.value)
  }

  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): (Double, Long, Long) = {
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

    // Train on batch
    model.fit(inputArray, labelsArray)

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

  def createValidationDataSetIterator(validationDataRDD: RDD[String], tokenizer: Tokenizer): DataSetIterator = {
    // Process the validation data to create features and labels
    val validationData = validationDataRDD.flatMap { text =>
      val tokens = tokenizer.encode(text)
      modelClass.createSlidingWindows(tokens).map { case (inputSeq, label) =>
        val inputArray = Nd4j.zeros(1, embeddingSize * windowSize)
        val labelArray = Nd4j.zeros(1, vocabularySize)

        // Convert input sequence and label to ND4J arrays
        val embedding = modelClass.createEmbeddingMatrix(inputSeq)
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
}
