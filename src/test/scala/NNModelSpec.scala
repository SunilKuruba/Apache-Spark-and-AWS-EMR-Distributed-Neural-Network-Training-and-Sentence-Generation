import com.typesafe.config.ConfigFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NNModelSpec extends AnyFlatSpec with Matchers {

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
}
