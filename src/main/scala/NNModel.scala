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

import java.io.{ByteArrayInputStream, ObjectInputStream, ObjectOutputStream}

class NNModel extends Serializable{
  val vocabularySize: Int = 3000
  val embeddingSize: Int = 32
  val windowSize: Int = 1
  val batchSize: Int = 16

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

  // Custom listener for collecting training metrics
  class CustomTrainingListener extends ScoreIterationListener(10) {
    private var currentScore: Double = 0.0

    override def iterationDone(model: org.deeplearning4j.nn.api.Model, iteration: Int, epochNum: Int): Unit = {
      super.iterationDone(model, iteration, epochNum)
      currentScore = model.score()
    }
  }

  class GradientNormListener(logFrequency: Int) extends IterationListener {
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

  def serializeModel(model: MultiLayerNetwork): Array[Byte] = {
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

  def deserializeModel(bytes: Array[Byte]): MultiLayerNetwork = {
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
}
