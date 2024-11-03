import org.slf4j.LoggerFactory

/**
 * Tokenizer class for converting words to indices and vice versa.
 * This is useful for preparing text data for neural network models.
 */
class Tokenizer extends Serializable {
  private val logger = LoggerFactory.getLogger(classOf[Tokenizer])

  // Maps to store word-index and index-word relationships
  // Given that the tokenizer runs for a short time, it is okay to have var(s) here
  var wordToIndex = Map[String, Int]()
  private var indexToWord = Map[Int, String]()
  private var currentIdx = 0

  /**
   * Fits the tokenizer on a given collection of texts.
   * It builds a vocabulary mapping each unique word to an index.
   *
   * @param texts The collection of texts to fit the tokenizer on.
   */
  def fit(texts: Seq[String]): Unit = {
    logger.info("Fitting the tokenizer to the provided texts.")

    texts.flatMap(_.split("\\s+")).distinct.foreach { word =>
      if (!wordToIndex.contains(word)) {
        wordToIndex += (word -> currentIdx)
        indexToWord += (currentIdx -> word)
        logger.debug(s"Added word '$word' with index $currentIdx to vocabulary.")
        currentIdx += 1
      }
    }
    logger.info(s"Tokenizer fitting complete. Vocabulary size: ${wordToIndex.size}")
  }

  /**
   * Encodes a given text into a sequence of indices.
   * If a word is not in the vocabulary, it defaults to index 0.
   *
   * @param text The text to encode.
   * @return A sequence of indices representing the words in the text.
   */
  def encode(text: String): Seq[Int] = {
    logger.debug(s"Encoding text: '$text'")
    val encoded = text.split("\\s+").map(word => wordToIndex.getOrElse(word, 0)).toSeq
    logger.debug(s"Encoded sequence: ${encoded.mkString(", ")}")
    encoded
  }

  /**
   * Decodes a sequence of indices back into a string of words.
   * If an index does not map to any word, it defaults to an empty string.
   *
   * @param indices The sequence of indices to decode.
   * @return The decoded string of words.
   */
  def decode(indices: Seq[Int]): String = {
    logger.debug(s"Decoding indices: ${indices.mkString(", ")}")
    val decoded = indices.map(idx => indexToWord.getOrElse(idx, "")).mkString(" ")
    logger.debug(s"Decoded text: '$decoded'")
    decoded
  }
}
