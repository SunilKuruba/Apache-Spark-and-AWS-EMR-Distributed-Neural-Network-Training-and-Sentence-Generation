trait FileSystem {
  def read(path: String): String  // Method to read from a file
  def write(path: String, data: String): Unit  // Method to write to a file
  def close(): Unit  // Method to close the file system connection
}

// Implement the LocalFileSystem class
class LocalFileSystem extends FileSystem {
  override def read(path: String): String = {
    // Implementation for reading from a local file
    scala.io.Source.fromFile(path).getLines().mkString("\n")
  }

  override def write(path: String, data: String): Unit = {
    // Implementation for writing to a local file
    import java.io.PrintWriter
    val writer = new PrintWriter(path)
    try {
      writer.write(data)
    } finally {
      writer.close()
    }
  }

  override def close(): Unit = {
    // Implementation to close any resources if needed
    println("LocalFileSystem closed.")
  }
}