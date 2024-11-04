import Main.Environment
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

@Ignore
class EndToEndIntegrationSpec extends AnyFlatSpec with Matchers {
  Main.environment = Environment.test;

  "End-to-end text generation" should "work with trained model" in {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("test"))
    try {
      // Training phase
      val textRDD = sc.parallelize(Seq(
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog jumps over the fox",
        "the lazy fox sleeps while the dog jumps"
      ))

      val outputStatsFilePath = "src/test/resources/output/e2e-metrics.csv";
      val metricsWriter = new LocalFileSystem(outputStatsFilePath)

      // Train the model
      val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
        .batchSizePerWorker(32)
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)
        .build()

      val train = new Train()
      val model = train.train(sc, textRDD, metricsWriter, 1, trainingMaster)

      // Text generation phase
      val tokenizer = new Tokenizer()
      tokenizer.fit(textRDD.collect())

      val textOutput = new TextOutput()
      val generatedText = textOutput.generateText(model, tokenizer, "the quick", 500)

      generatedText should not be empty
    } finally {
      sc.stop()
    }
  }

  "End-to-end program" should "work with param args" in {
    val seedToken = "new world"
    val inputPath = "src/test/resources/input/tiny_input.txt"
    val outputResultPath = "src/test/resources/output/output_result.txt"
    val outputStatsPath = "src/test/resources/output/output_statstics.csv"
    Main.main(Array(seedToken,inputPath, outputResultPath, outputStatsPath))
  }
}
