## DATA622 HW #5
- Assigned on October 25, 2018
- Due on November 28, 2018 11:59 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:

Read the following:
- [Apache Spark Python 101](https://www.datacamp.com/community/tutorials/apache-spark-python)
- [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)

Optional Readings:
- [Paper on RDD](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
- [Advanced Analytics with Spark: Patterns for Learning from Data at Scale, 2nd Edition](https://www.amazon.com/_/dp/1491972955), Chapters 1 - 2

Additional Resources:
- [Good intro resource on PySpark](https://annefou.github.io/pyspark/slides/spark/#1)
- [Spark The Definitive Guide](https://github.com/databricks/Spark-The-Definitive-Guide)
- [Google Cloud Dataproc, Spark on GCP](https://codelabs.developers.google.com/codelabs/cloud-dataproc-starter/)


### Critical Thinking (8 points total)

1. (2 points) How is the current Spark's framework different from MapReduce?  What are the tradeoffs of better performance speed in Spark?
Answer: MapReduce is a massively scalable, parallel processing framework which resides on the same physical nodes within the cluster. MapReduce and Spark are capable of efficiently processing massive volumes of both structured and unstructured data. MapReduce Hadoop has the fundamental flexibility to handle unstructured data regardless of the data source or native format. Additionally, disparate types of data stored on unrelated systems can all be deposited in the Hadoop cluster without the need to predetermine how the data will be queried. MapReduce Hadoop is designed to run batch jobs that address every file in the system. MapReduce is also ideal for scanning historical data and performing analytics where a short time-to-insight isn’t vital.

Spark was purposely designed to support in-memory processing. The net benefit of keeping everything in memory is the ability to perform iterative computations at blazing fast speeds. Along with supporting simple “map” and “reduce” operations, Spark supports SQL queries, streaming data, and complex analytics such as graph algorithms and machine learning. Since Spark runs on existing Hadoop clusters and is compatible with HDFS, HBase and any Hadoop storage system, users can combine all capabilities into a single workflow while accessing and processing all data in the current Hadoop environment.

Unlike MapReduce, Spark is designed for advanced, real-time analytics and has the framework and tools to deliver when shorter time-to-insight is critical. Included in Spark’s integrated framework are the Machine Learning Library (MLlib), the graph engine GraphX, the Spark Streaming analytics engine, and the real-time analytics tool, Shark. With this all-in-one platform, Spark is said to deliver greater consistency in product results across various types of analysis.

2. (2 points) Explain the difference between Spark RDD and Spark DataFrame/Datasets.

3. (1 point) Explain the difference between SparkML and Mahout.  

4. (1 point) Explain the difference between Spark.mllib and Spark.ml.

4. (2 points) Explain the tradeoffs between using Scala vs PySpark.


### Applied (7 points total)

Submit your Jupyter Notebook from following along with the code along in [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)
