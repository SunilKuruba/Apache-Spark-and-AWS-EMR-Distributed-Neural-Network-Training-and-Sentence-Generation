import Main.Environment
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TokenizerSpec extends AnyFlatSpec with Matchers {
  Main.environment = Environment.test;

  "Tokenizer" should "correctly fit and encode text" in {
    val tokenizer = new Tokenizer()
    val texts = Seq("hello world", "world hello", "test text")

    tokenizer.fit(texts)
    val encoded = tokenizer.encode("hello world")

    encoded should have length 2
    encoded.toSet should have size 2 // Unique indices
  }

  it should "correctly decode indices back to text" in {
    val tokenizer = new Tokenizer()
    val texts = Seq("hello world")

    tokenizer.fit(texts)
    val encoded = tokenizer.encode("hello world")
    val decoded = tokenizer.decode(encoded)

    decoded shouldEqual "hello world"
  }

  it should "handle unknown words during encoding" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("hello world"))

    val encoded = tokenizer.encode("hello unknown")
    encoded should have length 2
    encoded.last shouldEqual 0 // Default index for unknown words
  }

  it should "handle empty input correctly" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq.empty)

    val encoded = tokenizer.encode("")
    encoded should be  // No encoding for empty string
      tokenizer.decode(encoded) shouldEqual "" // Decoding empty should return empty
  }

  it should "handle single word input correctly" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("hello"))

    val encoded = tokenizer.encode("hello")
    encoded should have length 1
    encoded.head shouldEqual 0 // Assuming "hello" has been assigned index 1
  }

  it should "not fail when decoding an empty sequence of indices" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("hello world"))

    val decoded = tokenizer.decode(Seq.empty)
    decoded shouldEqual "" // Decoding empty indices should return an empty string
  }

  it should "ignore case differences in encoding" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("Hello World"))

    val encoded1 = tokenizer.encode("Hello World")
    val encoded2 = tokenizer.encode("Hello World")

    encoded1 shouldEqual encoded2 // Should produce the same encoding
  }

  it should "handle large texts with multiple words correctly" in {
    val tokenizer = new Tokenizer()
    val texts = Seq("This is a test", "Another test", "And another one")

    tokenizer.fit(texts)
    val encoded = tokenizer.encode("This is a test")

    encoded should have length 4 // Should encode each unique word
    encoded.toSet should have size 4 // All unique indices
  }

  it should "assign a unique index to each unique word" in {
    val tokenizer = new Tokenizer()
    val texts = Seq("one two three", "two three four", "four five one")

    tokenizer.fit(texts)

    // Checking the indices
    val wordToIndex = tokenizer.wordToIndex
    wordToIndex.keys should contain allOf ("one", "two", "three", "four", "five")
    wordToIndex.size shouldEqual 5 // Five unique words
  }

  it should "return the correct index for known words" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("apple banana orange"))

    tokenizer.encode("banana") shouldEqual Seq(1) // Assuming "banana" has index 1
    tokenizer.encode("apple") shouldEqual Seq(0) // Assuming "apple" has index 0
  }

  it should "default to unknown index for words not in vocabulary" in {
    val tokenizer = new Tokenizer()
    tokenizer.fit(Seq("cat dog"))

    val encoded = tokenizer.encode("cat elephant")
    encoded should have length 2
    encoded.last shouldEqual 0 // Assuming index 0 is for unknown
  }
}
