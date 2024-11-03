class Tokenizer extends Serializable {
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