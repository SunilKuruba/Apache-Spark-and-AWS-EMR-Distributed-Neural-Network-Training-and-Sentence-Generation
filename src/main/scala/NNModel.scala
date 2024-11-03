import com.typesafe.config.ConfigFactory
import org.apache.commons.io.output.ByteArrayOutputStream
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{ExponentialSchedule, ScheduleType}
import org.slf4j.LoggerFactory

import java.io.{ByteArrayInputStream, ObjectInputStream, ObjectOutputStream}

/**
 * Class representing a neural network model configuration and utilities.
 */
class NNModel extends Serializable {
  private val logger = LoggerFactory.getLogger(classOf[NNModel])
  private val config = ConfigFactory.load()

  // Configuration parameters from the config file
  private val vocabularySize: Int = config.getInt("model.vocabularySize")
  private val embeddingSize: Int = config.getInt("model.embeddingSize")
  private val windowSize: Int = config.getInt("model.windowSize")
  private val seed: Int = config.getInt("model.seed")
  private val initialValue: Double = config.getDouble("model.initialValue")
  private val gamma: Double = config.getDouble("model.gamma")
  private val layerSize: Int = config.getInt("model.layerSize")
  private val dropOut: Double = config.getDouble("model.dropOut")

  def buildModel(validationIterator: DataSetIterator): MultiLayerNetwork = {
    logger.info("Building the neural network model...")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, initialValue, gamma)))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(embeddingSize * windowSize)
        .nOut(layerSize)
        .activation(Activation.RELU)
        .dropOut(dropOut)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(layerSize)
        .nOut(layerSize)
        .activation(Activation.RELU)
        .dropOut(dropOut)
        .build())
      .layer(2, new OutputLayer.Builder()
        .nIn(layerSize)
        .nOut(vocabularySize)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new CustomTrainingListener, new GradientNormListener, new EvaluativeListener(validationIterator, 1))
    logger.info("Neural network model built successfully.")
    model
  }

  def createSlidingWindows(tokens: Seq[Int]): Seq[(Seq[Int], Int)] = {
    logger.info("Creating sliding windows for the input tokens.")
    tokens.sliding(windowSize + 1).map(window => (window.init, window.last)).toSeq
  }

  /**
   * Creates an embedding matrix for a given sequence using functional programming patterns.
   */
  def createEmbeddingMatrix(sequence: Seq[Int]): INDArray = {
    logger.info("Creating embedding matrix for the given sequence.")

    // Initialize base embedding matrix with random values
    val baseEmbedding = Nd4j.zeros(1, embeddingSize, sequence.length)

    // Function to create token embedding
    def createTokenEmbedding(): INDArray = Nd4j.randn(1, embeddingSize).mul(0.1)

    // Function to calculate positional encoding for a given position and dimension
    def calculatePositionalEncoding(pos: Int, dim: Int): Double = {
      val angle = pos / math.pow(10000, (2 * dim).toFloat / embeddingSize)
      if (dim % 2 == 0) math.sin(angle) else math.cos(angle)
    }

    // Apply token embeddings
    val withTokenEmbeddings = sequence.zipWithIndex.foldLeft(baseEmbedding) {
      case (embedding, (_, pos)) =>
        val tokenEmbedding = createTokenEmbedding()
        embedding.putSlice(pos, tokenEmbedding)
        embedding
    }

    // Apply positional encodings
    val finalEmbedding = sequence.indices.foldLeft(withTokenEmbeddings) { (embedding, pos) =>
      (0 until embeddingSize).foldLeft(embedding) { (emb, dim) =>
        val currentValue = emb.getDouble(0, dim, pos)
        val positionalValue = calculatePositionalEncoding(pos, dim)
        emb.putScalar(Array(0, dim, pos), currentValue + positionalValue)
        emb
      }
    }

    logger.info("Embedding matrix created successfully.")
    finalEmbedding
  }

  private class CustomTrainingListener extends ScoreIterationListener(10) {
    override def iterationDone(model: Model, iteration: Int, epochNum: Int): Unit = {
      super.iterationDone(model, iteration, epochNum)
      val currentScore: Double = model.score()
      logger.info(s"Iteration $iteration completed. Current score: $currentScore")
    }
  }

  private class GradientNormListener extends IterationListener {
    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      if (iteration % 10 == 0) {
        val gradients: INDArray = model.gradient().gradient()
        val stats = Seq(
          ("Mean", gradients.meanNumber().doubleValue()),
          ("Max", gradients.maxNumber().doubleValue()),
          ("Min", gradients.minNumber().doubleValue())
        )
        val statsString = stats.map { case (name, value) => s"$name = $value" }.mkString(", ")
        logger.info(s"Iteration $iteration: Gradient $statsString")
      }
    }
  }

  def serializeModel(model: MultiLayerNetwork): Array[Byte] = {
    logger.info("Serializing the model...")
    val baos = new ByteArrayOutputStream()
    try {
      val oos = new ObjectOutputStream(baos)
      oos.writeObject(model.params())
      oos.writeObject(model.getLayerWiseConfigurations)
      oos.close()
      logger.info("Model serialized successfully.")
      baos.toByteArray
    } finally {
      baos.close()
    }
  }

  def deserializeModel(bytes: Array[Byte]): MultiLayerNetwork = {
    logger.info("Deserializing the model...")
    val bais = new ByteArrayInputStream(bytes)
    try {
      val ois = new ObjectInputStream(bais)
      val params = ois.readObject().asInstanceOf[INDArray]
      val conf = ois.readObject().asInstanceOf[org.deeplearning4j.nn.conf.MultiLayerConfiguration]
      val model = new MultiLayerNetwork(conf)
      model.init()
      model.setParams(params)
      logger.info("Model deserialized successfully.")
      model
    } finally {
      bais.close()
    }
  }
}