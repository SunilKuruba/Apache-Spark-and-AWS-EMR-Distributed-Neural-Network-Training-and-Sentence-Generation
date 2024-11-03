import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TokenizerSpec extends AnyFlatSpec with Matchers {
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
}