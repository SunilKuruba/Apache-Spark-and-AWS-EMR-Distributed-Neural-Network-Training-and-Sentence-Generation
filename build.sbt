val scala2Version = "2.12.17"
lazy val sparkVersion = "3.5.3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "CS_441_Spark",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala2Version,
    scalaVersion := scala2Version,
    libraryDependencies ++= Seq(
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M1.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M1.1",
      "org.scala-lang.modules" %% "scala-collection-compat" % "2.8.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-streaming" % sparkVersion,
      "com.knuddels" % "jtokkit" % "0.6.1",
      "com.typesafe" % "config" % "1.4.3",
      "org.apache.mrunit" % "mrunit" % "1.1.0" % Test classifier "hadoop2",
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    assemblyMergeStrategy in assembly := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x => MergeStrategy.first
    }
  )