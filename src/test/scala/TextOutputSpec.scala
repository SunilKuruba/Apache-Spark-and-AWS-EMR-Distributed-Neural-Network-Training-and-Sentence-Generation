import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextOutputSpec extends AnyFlatSpec with Matchers {
  "TextOutput" should "generate text of correct length" in {
    val textOutput = new TextOutput()
    val tokenizer = new Tokenizer()
    val model = TestUtility.createDummyModel()

    tokenizer.fit(Seq("test text for generation"))
    val generated = textOutput.generateText(model, tokenizer, "test", 5)

    generated.split(" ") should have length 5
  }

  it should "handle different temperature values" in {
    val textOutput = new TextOutput()
    val tokenizer = new Tokenizer()
    val model = TestUtility.createDummyModel()

    tokenizer.fit(Seq("test text for generation"))

    val highTemp = textOutput.generateText(model, tokenizer, "test", 5, 1.0)
    val lowTemp = textOutput.generateText(model, tokenizer, "test", 5, 0.1)

    highTemp should not equal lowTemp
  }
}