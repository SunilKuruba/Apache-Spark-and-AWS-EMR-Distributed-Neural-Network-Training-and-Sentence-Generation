import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

object TestUtility {
   def createDummyDataIterator() = {
    import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
    import org.nd4j.linalg.dataset.DataSet

     val features = Nd4j.create(10, 100)
     val labels = Nd4j.create(10, 50)
    val dataSet = new DataSet(features, labels)

    new ListDataSetIterator(dataSet.asList(), 5)
  }

   def createDummyModel() = {
    val conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
      .seed(123)
      .list()
      .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
        .nIn(100)
        .nOut(50)
        .build())
      .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
        .nIn(50)
        .nOut(10)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }
}
