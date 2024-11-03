import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class TextOutput {
  def generateText(model: MultiLayerNetwork, tokenizer: Tokenizer, seedText: String, length: Int, temperature: Double = 0.7): String = {
    val embeddingSize: Int = 32
    val windowSize: Int = 1

    var currentSequence = tokenizer.encode(seedText).takeRight(windowSize)
    val generated = new ArrayBuffer[Int]()
    val rand = new Random()

    def sampleWithTemperature(arr: INDArray, temp: Double): Int = {
      val scaled = arr.div(temp)
      val expScaled = Transforms.exp(scaled)
      val prob = expScaled.div(expScaled.sum(1))

      // Convert to probabilities and sample
      val probArray = Array.ofDim[Double](prob.columns())
      for (i <- 0 until prob.columns()) {
        probArray(i) = prob.getDouble(Long.box(i))
      }

      // Sample using cumulative probabilities
      val cumSum = probArray.scanLeft(0.0)(_ + _).tail
      val sample = rand.nextDouble()
      cumSum.zipWithIndex.find(_._1 >= sample).map(_._2).getOrElse(0)
    }

    for (_ <- 1 to length) {
      val embedding = new LLMModel().createEmbeddingMatrix(currentSequence)
      val attentionOutput = new Train().selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)

      val output = model.output(flattenedAttention)
      val nextTokenIndex = sampleWithTemperature(output, temperature)

      generated += nextTokenIndex
      currentSequence = (currentSequence.tail :+ nextTokenIndex).takeRight(windowSize)
    }

    tokenizer.decode(generated)
  }
}