import org.apache.spark.SparkContext
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

// Define a trait representing a generic file system interface
trait FileSystem {
  // Method to create a new file or setup file-related resources
  def create(): Unit

  // Method to read the content from a file and return it as a String
  def read(): String

  // Method to write data to a file
  def write(data: String): Unit

  // Method to close any open resources or finalize writing operations
  def close(): Unit
}

// Implement a file system that works with the local file system
class LocalFileSystem(path: String) extends FileSystem {
  // StringBuilder to accumulate data to be written to the file
  private val stringBuilder = new StringBuilder()

  /**
   * Creates a new file at the specified path if it does not already exist.
   */
  override def create(): Unit = {
    val filePath = Paths.get(path)
    val parentDir = filePath.getParent
    if (!Files.exists(parentDir)) Files.createDirectories(parentDir)
    if (!Files.exists(filePath)) Files.createFile(filePath)
  }

  /**
   * Reads the content of the file at the specified path and returns it as a String.
   */
  override def read(): String = {
    Files.readString(Paths.get(path))
  }

  /**
   * Appends the given data to the internal StringBuilder.
   * The actual file writing is deferred until `close()` is called.
   */
  override def write(data: String): Unit = {
    stringBuilder.append(data)
  }

  /**
   * Writes the accumulated data from the StringBuilder to the file at the specified path.
   */
  override def close(): Unit = {
    Files.write(Paths.get(path), stringBuilder.toString().getBytes(StandardCharsets.UTF_8))
  }
}

// Implement a file system that works with Amazon S3 using Spark
class S3FileSystem(sparkContext: SparkContext, path: String) extends FileSystem {
  // StringBuilder to accumulate data to be written to the S3 path
  private val stringBuilder = new StringBuilder()

  /**
   * Creates an empty file in the S3 path using Spark.
   * This is done by saving an empty RDD to the specified path.
   */
  override def create(): Unit = {
    sparkContext.emptyRDD[String].coalesce(1).saveAsTextFile(path.substring(0, path.lastIndexOf("/")))
  }

  /**
   * Reads the content from the S3 path using Spark and returns it as a String.
   * The data is collected from the RDD and concatenated with newline characters.
   */
  override def read(): String = {
    sparkContext.textFile(path).collect().mkString("\n")
  }

  /**
   * Appends the given data to the internal StringBuilder.
   * The actual writing to S3 is deferred until `close()` is called.
   */
  override def write(data: String): Unit = {
    stringBuilder.append(data)
  }

  /**
   * Writes the accumulated data from the StringBuilder to the S3 path using Spark.
   * The data is parallelized and saved as a text file.
   */
  override def close(): Unit = {
    sparkContext.parallelize(Seq(stringBuilder.toString())).coalesce(1).saveAsTextFile(path)
  }
}
