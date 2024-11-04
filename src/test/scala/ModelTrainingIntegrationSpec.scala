import Main.Environment
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ModelTrainingIntegrationSpec extends AnyFlatSpec with Matchers {
  Main.environment = Environment.test;

  "Complete training pipeline" should "successfully train model" in {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("test"))
    try {
      val textRDD = sc.parallelize(Seq(
        "this is a test text",
        "another test text",
        "more sample data"
      ))

      val outputStatsFilePath = "src/test/resources/output/test-metrics.csv";
      val metricsWriter = new LocalFileSystem(outputStatsFilePath)

      // Train the model
      val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
        .batchSizePerWorker(32)
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)
        .build()

      val train = new Train()
      val model = train.train(sc, textRDD, metricsWriter, 1, trainingMaster)

      model shouldBe a [MultiLayerNetwork]
      java.nio.file.Files.exists(java.nio.file.Paths.get("src/test/resources/output/test-metrics.csv")) shouldBe true

    } finally {
      sc.stop()
    }
  }
}