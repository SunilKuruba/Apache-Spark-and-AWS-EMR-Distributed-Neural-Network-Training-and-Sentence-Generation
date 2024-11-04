# Spark-LLM-AWS-EMR: Distributed Neural Network Training and Sentence Prediction

**Author**: Sunil Kuruba <br />
**UIN**: 659375633 <br />
**Email**: skuru@uic.edu <br />
**Instructor**: Mark Grechanik

Youtube video - TBD

## Description
This project showcases a scalable training pipeline for a neural network leveraging Apache Spark for distributed data processing and deep learning capabilities. The `Train` class is designed for comprehensive model training with built-in support for self-attention mechanisms, enhancing the embedding process for better model performance. Key aspects include reading configurations, handling data preprocessing with Spark RDDs, and training over multiple epochs with detailed metric tracking. The self-attention implementation computes query, key, and value matrices to enrich the representation learning phase. The main execution logic, contained in the `Main` object, sets up the environment, manages data input from local or cloud-based sources like S3, and writes results and training metrics. Integration with DL4J (`MultiLayerNetwork`, `ParameterAveragingTrainingMaster`) and ND4J (`INDArray`, `Transforms`) facilitates model configuration and parameter optimization, allowing for seamless execution in distributed settings. The code demonstrates an efficient blend of Spark-based parallelism with deep learning, making it suitable for large-scale NLP and machine learning tasks.
## Project Structure

```bash
├── src
│   ├── main
│   │   ├── scala
│   │   │   ├── FileSystem.scala           # Module for file operations (read/write)
│   │   │   ├── Main.scala                 # Main entry point for the application
│   │   │   ├── NNModel.scala              # Neural network model and architecture
│   │   │   ├── TextOutput.scala           # Handles the text output formatting
│   │   │   ├── Tokenizer.scala            # Tokenization logic for input data
│   │   │   ├── Train.scala                # Module to train the LLM model
│   │   ├── resources
│   │   │   ├── input
│   │   │   ├── output
│   │   │   ├── application.conf           # Configuration file for project settings
│   ├── test
│   │   ├── scala
│   │   │   ├── EndToEndIntegrationSpec.scala      # End-to-end integration tests
│   │   │   ├── ModelTrainingIntegrationSpec.scala # Integration tests for model training
│   │   │   ├── NNModelSpec.scala                  # Unit tests for the NN model
│   │   │   ├── TestUtility.scala                  # Utility functions for testing
│   │   │   ├── TokenizerSpec.scala                # Unit tests for the tokenizer logic
│   │   ├── resources
│   │   │   ├── input
│   │   │   ├── output
├── target                                 # Compiled files and build output
├── .gitignore                             # Git ignore file
├── build.sbt                              # Build configuration file for SBT
└── README.md                              # Project documentation
```

## Prerequisites

1. Apache Spark (version 3.x recommended)
2. Scala (version 2.12 or compatible)
3. AWS EMR cluster setup and configured
4. AWS S3 setup and configured
4. SBT (Scala Build Tool) for building the project
5. Java Development Kit (JDK) 8 or higher

## Steps to Execute the Project

### 1. Setup the Project Environment
- Clone the repository:
  ```bash
  git clone https://github.com/SunilKuruba/Spark-LLM-AWS-EMR.git
  cd CS_441_spark
  ```
- Compile the project using SBT:
  ```bash
  sbt clean update compile
  ```
### 2. Prepare AWS
* Create S3 bucket to store the JARs and input data. Make a note of the S3 paths.
* Load the input training files of your choice in the S3 folder
```bash
├── jar
│   └── spark_hw2.jar                 # Executable jar file
├── input
│   ├── train.txt                     # Input text file               # Input CSV file
├── output                            # Output folder (initially empty)
```

### 2. Deploy to AWS EMR
- Ensure that your AWS EMR cluster is up and running.
- Package your project into a `.jar` file:
  ```bash
  sbt package
  ```
- Upload the `.jar` file to S3:
  ```bash
  aws s3 cp target/scala-2.12/CS_441_Spark-assembly-0.1.jar s3://<your-bucket-name>/
  ```
- Run the Spark job on your EMR cluster:

### 3. Input Data Preparation
- Prepare input text files under `src/main/resources/input/` or upload them to an accessible location in S3 input folder.

### 4. Run the Training
- Train the model by running the `Train` module

## Testing
Run the tests using SBT:
```bash
sbt test
```

## Results
After training, the model should generate coherent sentences from a seed input, showcasing text generation capabilities similar to GPT models.
```