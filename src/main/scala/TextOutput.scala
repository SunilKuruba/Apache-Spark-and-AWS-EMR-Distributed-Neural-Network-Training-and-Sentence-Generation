import com.typesafe.config.ConfigFactory
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.LoggerFactory

/**
 * Class responsible for generating text from a trained neural network model.
 */
class TextOutput {
  private val logger = LoggerFactory.getLogger(classOf[TextOutput])

  /**
   * Generates text based on a given seed text using the trained model.
   *
   * @param model       The trained MultiLayerNetwork model.
   * @param tokenizer   The Tokenizer used to encode and decode text.
   * @param seedText    The initial text to start generating from.
   * @param length      The number of tokens to generate.
   * @param temperature The temperature parameter for controlling randomness.
   * @return The generated text as a String.
   */
  def generateText(
                    model: MultiLayerNetwork,
                    tokenizer: Tokenizer,
                    seedText: String,
                    length: Int,
                    temperature: Double = 0.7
                  ): String = {
    logger.info(s"Generating text with seed: '$seedText', length: $length, and temperature: $temperature")

    val config = ConfigFactory.load()
    val embeddingSize: Int = config.getInt("model.embeddingSize")
    val windowSize: Int = config.getInt("model.windowSize")
    val initialSequence = tokenizer.encode(seedText).takeRight(windowSize)

    /**
     * Samples a token index using the provided temperature for randomness control.
     *
     * @param arr The output array from the model.
     * @param temp The temperature value to scale probabilities.
     * @return The sampled token index.
     */
    def sampleWithTemperature(arr: INDArray, temp: Double): Int = {
      logger.debug("Sampling next token with temperature scaling")
      val scaled = arr.div(temp)
      val expScaled = Transforms.exp(scaled)
      val prob = expScaled.div(expScaled.sum(1))

      val probArray = (0 until prob.columns()).map(i => prob.getDouble(Long.box(i))).toArray
      val cumSum = probArray.scanLeft(0.0)(_ + _).tail
      val sample = scala.util.Random.nextDouble()

      cumSum.zipWithIndex
        .find(_._1 >= sample)
        .map(_._2)
        .getOrElse(0)
    }

    /**
     * Generates the next token given a sequence of tokens.
     *
     * @param currentSeq The current sequence of tokens.
     * @return The next token index.
     */
    def generateNextToken(currentSeq: Seq[Int]): Int = {
      logger.debug(s"Current sequence: ${currentSeq.mkString(", ")}")

      val embedding = new NNModel().createEmbeddingMatrix(currentSeq)
      val attentionOutput = new Train().selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)

      val output = model.output(flattenedAttention)
      sampleWithTemperature(output, temperature)
    }

    /**
     * Recursively generates tokens until the desired length is reached.
     *
     * @param sequence The current sequence of tokens.
     * @param generatedTokens The tokens generated so far.
     * @param remaining The number of tokens still to generate.
     * @return The complete list of generated tokens.
     */
    def generateTokens(
                        sequence: Seq[Int],
                        generatedTokens: List[Int],
                        remaining: Int
                      ): List[Int] = {
      if (remaining <= 0) generatedTokens.reverse
      else {
        val nextToken = generateNextToken(sequence)
        val nextSequence = (sequence.tail :+ nextToken).takeRight(windowSize)
        generateTokens(nextSequence, nextToken :: generatedTokens, remaining - 1)
      }
    }

    val generatedTokens = generateTokens(initialSequence, List.empty, length)
    val generatedText = tokenizer.decode(generatedTokens)

    logger.info(s"Generated text: $generatedText")
    generatedText
  }
}