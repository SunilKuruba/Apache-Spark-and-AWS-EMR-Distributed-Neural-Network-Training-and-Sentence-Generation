import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable.ArrayBuffer

class SentenceGeneration extends Serializable {
  val vocabularySize: Int = 1000
  val embeddingSize: Int = 64
  val windowSize: Int = 5
  val batchSize: Int = 320

  // Tokenizer class for handling text conversion
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
      indices.map(idx => indexToWord.getOrElse(idx, "<UNK>")).mkString(" ")
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

  // Build neural network model
  def buildModel(): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .updater(new Adam(0.001))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(embeddingSize * windowSize)
        .nOut(512)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(512)
        .nOut(256)
        .activation(Activation.RELU)
        .build())
      .layer(2, new OutputLayer.Builder()
        .nIn(256)
        .nOut(vocabularySize)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))
    model
  }

  // Train the model using Spark RDD
  def train(sc: SparkContext, textRDD: RDD[String], epochs: Int): MultiLayerNetwork = {
    val tokenizer = new SimpleTokenizer()

    // Fit tokenizer and broadcast
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    // Create and broadcast model
    val model = buildModel()
    val broadcastModel = sc.broadcast(model)

    for (epoch <- 1 to epochs) {
      println(s"Training epoch $epoch")

      // Create training samples using sliding window
      val samplesRDD = textRDD.flatMap { text =>
        val tokens = broadcastTokenizer.value.encode(text)
        createSlidingWindows(tokens)
      }

      // Process batches
      samplesRDD.foreachPartition { partition =>
        val localModel = broadcastModel.value
        val batchBuffer = new ArrayBuffer[(Seq[Int], Int)]()

        partition.foreach { sample =>
          batchBuffer += sample
          if (batchBuffer.size >= batchSize) {
            processBatch(localModel, batchBuffer.toSeq)
            batchBuffer.clear()
          }
        }

        // Process remaining samples
        if (batchBuffer.nonEmpty) {
          processBatch(localModel, batchBuffer.toSeq)
        }
      }
    }

    model
  }

  // Process a batch of samples
  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): Unit = {
    val inputArray = Nd4j.zeros(batch.size, embeddingSize * windowSize)
    val labelsArray = Nd4j.zeros(batch.size, vocabularySize)

    batch.zipWithIndex.foreach { case ((sequence, label), idx) =>
      val embedding = createEmbeddingMatrix(sequence)
      val attentionOutput = selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
      inputArray.putRow(idx, flattenedAttention)
      labelsArray.putScalar(Array(idx, label), 1.0)
    }

    model.fit(inputArray, labelsArray)
  }

  // Generate text using the trained model
  def generateText(model: MultiLayerNetwork, tokenizer: SimpleTokenizer, seedText: String, length: Int): String = {
    var currentSequence = tokenizer.encode(seedText).takeRight(windowSize)
    val generated = new ArrayBuffer[Int]()

    for (_ <- 1 to length) {
      val embedding = createEmbeddingMatrix(currentSequence)
      val attentionOutput = selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)

      val output = model.output(flattenedAttention)
      val nextTokenIndex = Nd4j.argMax(output, 1).getInt(0)

      generated += nextTokenIndex
      currentSequence = (currentSequence.tail :+ nextTokenIndex).takeRight(windowSize)
    }

    tokenizer.decode(generated)
  }
}

// Usage example
object SimpleLanguageModelExample {
  def main2(args: Array[String]): Unit = {
    val sc = new SparkContext("local[*]", "SimpleLanguageModel")

    // Sample text data
    val texts = Seq(
      "the quick brown fox jumps over the lazy dog",
      "to be or not to be that is the question",
      // Add more training texts here
    )
    val textRDD = sc.parallelize(texts)

    // Create and train model
    val model = new SentenceGeneration()
    val trainedModel = model.train(sc, textRDD, 10)

    // Generate text
    val tokenizer = new model.SimpleTokenizer()
    tokenizer.fit(texts)
    val generatedText = model.generateText(trainedModel, tokenizer, "quick brown fox", 5)
    println(s"Generated text: $generatedText")

    sc.stop()
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SimpleLanguageModel").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Define the path to the text file
    val filePath = "src/main/resources/tiny_input.txt" // Replace with the actual file path

    // Load the text file into an RDD, where each line in the file is a separate RDD element
    val textRDD = sc.textFile(filePath)
      .map(_.trim)                         // Trim whitespace
      .filter(line => line.nonEmpty)       // Remove empty lines
      .cache()

    // Create and train the model
    val model = new SentenceGeneration()

    // Assuming 'train' method takes an RDD and the number of epochs as arguments
    val trainedModel = model.train(sc, textRDD, 1)

    // Correctly initialize and use the tokenizer
    val tokenizer = new model.SimpleTokenizer()
    val texts = textRDD.collect() // Collect the RDD to use with the tokenizer
    tokenizer.fit(texts) // Ensure 'fit' works with collected texts

    // Generate text using the trained model
    val generatedText = model.generateText(trainedModel, tokenizer, "scientist", 5)
    println(s"Generated text: $generatedText")

    //      val generatedText2 = model.generateText(
    //        trainedModel,
    //        tokenizer,
    //        seedText = "quick brown fox",
    //        length = 5,
    //        temperature = temp
    //      )

    sc.stop()
  }

}