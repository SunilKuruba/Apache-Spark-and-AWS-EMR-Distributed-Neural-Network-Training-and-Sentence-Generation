import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

/**
 * A trait representing a generic file system interface.
 *
 * This trait provides a contract for file system operations, including
 * creating, reading, writing, and closing files. Any concrete implementation
 * of this trait should provide the actual logic for these operations.
 *
 * This trait extends Serializable to ensure that implementations can be
 * serialized, which is often required for distributed systems or saving
 * state across sessions.
 */
trait FileSystem extends Serializable {

  /**
   * Creates a new file or sets up necessary resources for file-related operations.
   *
   * Implementations should define how a file is created or initialized,
   * including any pre-requisites or configurations required for file access.
   */
  def create(): Unit

  /**
   * Reads the content from a file and returns it as a String.
   *
   * This method should handle the process of opening the file, reading its
   * content, and returning the data in a String format. Implementations should
   * consider error handling for scenarios where the file may not exist or
   * is not readable.
   *
   * @return The content of the file as a String.
   * @throws IOException If there is an error reading the file.
   */
  def read(): String

  /**
   * Writes data to a file.
   *
   * This method accepts a String input and should handle the logic for
   * writing that data to the file. Implementations should ensure that
   * the file is properly opened for writing and that any existing content
   * is handled according to the desired behavior (e.g., overwrite or append).
   *
   * @param data The data to be written to the file.
   * @throws IOException If there is an error writing to the file.
   */
  def write(data: String): Unit

  /**
   * Closes any open resources or finalizes writing operations.
   *
   * This method should be called to release any resources associated with
   * the file operations, such as closing file handles or flushing buffers.
   * Implementations should ensure that all resources are cleaned up properly
   * to prevent memory leaks.
   */
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
  private val logger = LoggerFactory.getLogger(this.getClass)

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
    try {
      sparkContext.parallelize(Seq(stringBuilder.toString())).coalesce(1).saveAsTextFile(path)
    }catch {
      case e: Exception =>
        logger.error("Error occurred while writing metrics or reading input data.", e)
    }
  }
}
