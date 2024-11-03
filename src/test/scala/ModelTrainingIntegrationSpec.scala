import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ModelTrainingIntegrationSpec extends AnyFlatSpec with Matchers {
  "Complete training pipeline" should "successfully train model" in {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("test"))
    try {
      val textRDD = sc.parallelize(Seq(
        "this is a test text",
        "another test text",
        "more sample data"
      ))

      val train = new Train()
      val metricsWriter = new java.io.BufferedWriter(
        new java.io.FileWriter("src/test/resources/output/test-metrics.csv")
      )

      val model = train.train(sc, textRDD, metricsWriter, 1)

      model shouldBe a [MultiLayerNetwork]
      java.nio.file.Files.exists(java.nio.file.Paths.get("src/test/resources/output/test-metrics.csv")) shouldBe true

    } finally {
      sc.stop()
    }
  }
}