import com.typesafe.config.ConfigFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NNModelSpec extends AnyFlatSpec with Matchers {
  "NNModel" should "create embedding matrix with correct dimensions" in {
    val model = new NNModel()
    val sequence = Seq(1)

    val embedding = model.createEmbeddingMatrix(sequence)

    embedding.shape()(0) shouldEqual 1
    embedding.shape()(1) shouldEqual ConfigFactory.load().getInt("model.embeddingSize")
    embedding.shape()(2) shouldEqual sequence.length
  }

  it should "build model with correct architecture" in {
    val model = new NNModel()
    val dataIterator = TestUtility.createDummyDataIterator()

    val network = model.buildModel(dataIterator)

    network.getLayerWiseConfigurations.getConfs should have length 3
  }

  it should "correctly serialize and deserialize model" in {
    val model = new NNModel()
    val network = model.buildModel(TestUtility.createDummyDataIterator())

    val serialized = model.serializeModel(network)
    val deserialized = model.deserializeModel(serialized)

    deserialized.getLayerWiseConfigurations.toJson shouldEqual
      network.getLayerWiseConfigurations.toJson
  }

    it should "correctly handle varying sequence lengths in embedding matrix" in {
      val model = new NNModel()
      val sequence1 = Seq(1)
      val sequence2 = Seq(3)

      val embedding1 = model.createEmbeddingMatrix(sequence1)
      val embedding2 = model.createEmbeddingMatrix(sequence2)

      embedding1.shape()(0) shouldEqual 1
      embedding1.shape()(1) shouldEqual ConfigFactory.load().getInt("model.embeddingSize")
      embedding1.shape()(2) shouldEqual sequence1.length

      embedding2.shape()(0) shouldEqual 1
      embedding2.shape()(1) shouldEqual ConfigFactory.load().getInt("model.embeddingSize")
      embedding2.shape()(2) shouldEqual sequence2.length
    }
}
