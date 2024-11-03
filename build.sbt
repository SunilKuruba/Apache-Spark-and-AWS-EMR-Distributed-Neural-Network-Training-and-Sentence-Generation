val scala2Version = "2.12.17"
lazy val sparkVersion = "3.5.3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "CS_441_Spark",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala2Version,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-streaming" % sparkVersion,

      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M1.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M1.1",

      "org.scala-lang.modules" %% "scala-collection-compat" % "2.8.1",
      "com.knuddels" % "jtokkit" % "1.1.0",
      "com.typesafe" % "config" % "1.4.3",

      "org.apache.mrunit" % "mrunit" % "1.1.0" % Test classifier "hadoop2",
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "org.scalatest" %% "scalatest" % "3.2.15" % Test,
      "org.mockito" %% "mockito-scala" % "1.17.12" % Test
    ),
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) =>
        xs match {
          case "MANIFEST.MF" :: Nil =>   MergeStrategy.discard
          case "services" ::_       =>   MergeStrategy.concat
          case _                    =>   MergeStrategy.discard
        }
      case "reference.conf"  => MergeStrategy.concat
      case x if x.endsWith(".proto") => MergeStrategy.rename
      case x if x.contains("hadoop") => MergeStrategy.first
      case  _ => MergeStrategy.first
    }
  )

resolvers += "Conjars Repo" at "https://conjars.org/repo"