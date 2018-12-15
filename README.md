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

Answer: Spark RDD APIs – An RDD stands for Resilient Distributed Datasets. It is Read-only partition collection of records. RDD is the fundamental data structure of Spark. It allows a programmer to perform in-memory computations on large clusters in a fault-tolerant manner. RDD is a distributed collection of data elements spread across many machines in the cluster. RDDs are a set of Java or Scala objects representing data.It can easily and efficiently process data which is structured as well as unstructured. But like Dataframe and DataSets, RDD does not infer the schema of the ingested data and requires the user to specify it. Data source API allows that an RDD could come from any data source e.g. text file, database via JDBC etc. and easily handle data with no predefined structure.
RDDs contains the collection of records which are partitioned. The basic unit of parallelism in an RDD is called partition. Each partition is one logical division of data which is immutable and created through some transformation on existing partitions. Immutability helps to achieve consistency in computations. We can move from RDD to DataFrame (If RDD is in tabular format) by toDF() method or we can do the reverse by the .rdd method. RDD provides a familiar object-oriented programming style with compile-time type safety.No inbuilt optimization engine is available in RDD. When working with structured data, RDDs cannot take advantages of sparks advance optimizers. For example, catalyst optimizer and Tungsten execution engine. Developers optimise each RDD on the basis of its attributes. Whenever Spark needs to distribute the data within the cluster or write the data to disk, it does so use Java serialization. The overhead of serializing individual Java and Scala objects is expensive and requires sending both data and structure between nodes. There is overhead for garbage collection that results from creating and destroying individual objects. Efficiency is decreased when serialization is performed individually on a java and scala object which takes lots of time. Spark evaluates RDDs lazily. They do not compute their result right away. Instead, they just remember the transformation applied to some base data set. Spark compute Transformations only when an action needs a result to sent to the driver program. RDD APIs are available in Java, Scala, Python, and R languages. Hence, this feature provides flexibility to the developers. In RDD APIs use schema projection is used explicitly. Hence, we need to define the schema (manually). RDD API is slower to perform simple grouping and aggregation operations.

Spark Dataframe APIs – Unlike an RDD, data organized into named columns. For example a table in a relational database. It is an immutable distributed collection of data. DataFrame in Spark allows developers to impose a structure onto a distributed collection of data, allowing higher-level abstraction. A DataFrame is a distributed collection of data organized into named columns. It is conceptually equal to a table in a relational database. It can process structured and unstructured data efficiently. It organizes the data in the named column. DataFrames allow the Spark to manage schema. Data source API allows Data processing in different formats (AVRO, CSV, JSON, and storage system HDFS, HIVE tables, MySQL). It can read and write from various data sources that are mentioned above. After transforming into DataFrame one cannot regenerate a domain object. For example, if you generate testDF from testRDD, then you won’t be able to recover the original RDD of the test class. If you are trying to access the column which does not exist in the table in such case Dataframe APIs does not support compile-time error. It detects attribute error only at runtime. Optimization takes place using catalyst optimizer. Dataframes use catalyst tree transformation framework in four phases: a) Analyzing a logical plan to resolve references. b) Logical plan optimization. c) Physical planning. d) Code generation to compile parts of the query to Java bytecode. Spark DataFrame Can serialize the data into off-heap storage (in memory) in binary format and then perform many transformations directly on this off heap memory because spark understands the schema. There is no need to use java serialization to encode the data. It provides a Tungsten physical execution backend which explicitly manages memory and dynamically generates bytecode for expression evaluation. Avoids the garbage collection costs in constructing individual objects for each row in the dataset. Use of off heap memory for serialization reduces the overhead. It generates byte code dynamically so that many operations can be performed on that serialized data. No need for deserialization for small operations. Spark evaluates DataFrame lazily, that means computation happens only when action appears (like display result, save output). It also has APIs in the different languages like Java, Python, Scala, and R. Auto-discovering the schema from the files and exposing them as tables through the Hive Meta store. We did this to connect standard SQL clients to our engine. And explore our dataset without defining the schema of our files. DataFrame API is very easy to use. It is faster for exploratory analysis, creating aggregated statistics on large data sets.


Spark Dataset APIs – Datasets in Apache Spark are an extension of DataFrame API which provides type-safe, object-oriented programming interface. Dataset takes advantage of Spark’s Catalyst optimizer by exposing expressions and data fields to a query planner. It is an extension of DataFrame API that provides the functionality of – type-safe, object-oriented programming interface of the RDD API and performance benefits of the Catalyst query optimizer and off heap storage mechanism of a DataFrame API. It also efficiently processes structured and unstructured data. It represents data in the form of JVM objects of row or a collection of row object. Which is represented in tabular forms through encoders. Dataset API of spark also support data from different sources. It overcomes the limitation of DataFrame to regenerate the RDD from Dataframe. Datasets allow you to convert your existing RDD and DataFrames into Datasets. It provides compile-time type safety. It includes the concept of Dataframe Catalyst optimizer for optimizing query plan. When it comes to serializing data, the Dataset API in Spark has the concept of an encoder which handles conversion between JVM objects to tabular representation. It stores tabular representation using spark internal Tungsten binary format. Dataset allows performing the operation on serialized data and improving memory use. It allows on-demand access to individual attribute without desterilizing the entire object. There is also no need for the garbage collector to destroy object because serialization takes place through Tungsten. That uses off heap data serialization. It allows performing an operation on serialized data and improving memory use. Thus it allows on-demand access to individual attribute without deserializing the entire object. It also evaluates lazily as RDD and Dataset. Dataset APIs is currently only available in Scala and Java. Spark version 2.1.1 does not support Python and R. Auto discover the schema of the files because of using Spark SQL engine. In Dataset it is faster to perform aggregation operation on plenty of data sets.

From the comparison between RDD vs DataFrame vs Dataset, it is clear when to use RDD or DataFrame and/or Dataset. As a result, RDD offers low-level functionality and control. The DataFrame and Dataset allow custom view and structure. It offers high-level domain-specific operations, saves space, and executes at high speed. Select one out of DataFrames and/or Dataset or RDDs APIs, that meets your needs and play with Spark.

Referrence: https://data-flair.training/blogs/apache-spark-rdd-vs-dataframe-vs-dataset/ 

3. (1 point) Explain the difference between SparkML and Mahout.  

Answer: Mahout has proven capabilities that Spark’s MlLib still haven't touched.  While Mahout is mature and comes with many ML algorithms to choose from, it is built atop MapReduce, and therefore constrained by disk accesses it is slow and does not handle iterative jobs very well. Since Machine Learning algorithms generally use many iterations, this makes Mahout run very slowly. In contrast, MlLib is built on top of Spark, making it much faster than Mahout.

The main difference will come from underlying frameworks. In case of Mahout it is Hadoop MapReduce and in case of MLib it is Spark. To be more specific - from the difference in per job overhead If your ML algorithm mapped to the single MR job - main difference will be only startup overhead, which is dozens of seconds for Hadoop MR, and let say 1 second for Spark. So in case of model training it is not that important.Things will be different if your algorithm is mapped to many jobs. In this case we will have the same difference on overhead per iteration and it can be game changer. Lets assume that we need 100 iterations, each needed 5 seconds of cluster CPU.

On Spark: it will take 100*5 + 100*1 seconds = 600 seconds.
On Hadoop: MR (Mahout) it will take 100*5+100*30 = 3500 seconds.

In the same time Hadoop MR is much more mature framework then Spark and if you have a lot of data, and stability is paramount - I would consider Mahout as serious alternative

MLlib is a loose collection of high-level algorithms that runs on Spark. This is what Mahout used to be only Mahout of old was on Hadoop Mapreduce. In 2014 Mahout announced it would no longer accept Hadoop Mapreduce code and completely switched new development to Spark (with other engines possibly in the offing, like H2O).

The most significant thing to come out of this is a Scala-based generalized distributed optimized linear algebra engine and environment including an interactive Scala shell. Perhaps the most important word is "generalized". Since it runs on Spark anything available in MLlib can be used with the linear algebra engine of Mahout-Spark.

If you need a general engine that will do a lot of what tools like R do but on really big data, look at Mahout. If you need a specific algorithm, look at each to see what they have. For instance Kmeans runs in MLlib but if you need to cluster A'A (a cooccurrence matrix used in recommenders) you'll need them both because MLlib doesn't have a matrix transpose or A'A (actually Mahout does a thin-optimized A'A so the transpose is optimized out).

Mahout also includes some innovative recommender building blocks that offer things found in no other OSS. Mahout still has its older Hadoop algorithms but as fast compute engines like Spark become the norm most people will invest there.

https://www.linkedin.com/pulse/choosing-machine-learning-frameworks-apache-mahout-vs-debajani/
https://stackoverflow.com/questions/23511459/what-is-the-difference-between-apache-mahout-and-apache-sparks-mllib

4. (1 point) Explain the difference between Spark.mllib and Spark.ml.

Answer: MLlib is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. At a high level, it provides tools such as:
ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
Featurization: feature extraction, transformation, dimensionality reduction, and selection
Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
Persistence: saving and load algorithms, models, and Pipelines
Utilities: linear algebra, statistics, data handling, etc.

“Spark ML” is not an official name but occasionally used to refer to the MLlib DataFrame-based API. This is majorly due to the org.apache.spark.ml Scala package name used by the DataFrame-based API, and the “Spark ML Pipelines” term we used initially to emphasize the pipeline concept. 

https://spark.apache.org/docs/latest/ml-guide.html

4. (2 points) Explain the tradeoffs between using Scala vs PySpark.
Answer: 
Performance - Scala is frequently over 10 times faster than Python. Scala uses Java Virtual Machine (JVM) during runtime which gives is some speed over Python in most cases. Python is dynamically typed and this reduces the speed. Compiled languages are faster than interpreted. In case of Python, Spark libraries are called which require a lot of code processing and hence slower performance.

Learning Curve - Both are functional and object oriented languages which have similar syntax in addition to a thriving support communities. Scala may be a bit more complex to learn in comparison to Python due to its high-level functional features. Python is preferable for simple intuitive logic whereas Scala is more useful for complex workflows. 

Concurrency - Scala has multiple standard libraries and cores which allows quick integration of the databases in Big Data ecosystems. Scala allows writing of code with multiple concurrency primitives whereas Python doesn’t support concurrency or multithreading. Due to its concurrency feature, Scala allows better memory management and data processing. However Python does support heavyweight process forking. Here, only one thread is active at a time. So whenever a new code is deployed, more processes must be restarted which increases the memory overhead.

Usability - Both are expressive and we can achieve high functionality level with them. Python is more user friendly and concise. Scala is always more powerful in terms of framework, libraries, implicit, macros etc. Scala works well within the MapReduce framework because of its functional nature. Many Scala data frameworks follow similar abstract data types that are consistent with Scala’s collection of APIs. Developers just need to learn the basic standard collections, which allow them to easily get acquainted with other libraries. Spark is written in Scala so knowing Scala will let you understand and modify what Spark does internally. Moreover many upcoming features will first have their APIs in Scala and Java and the Python APIs evolve in the later versions. But for NLP, Python is preferred as Scala doesn’t have many tools for machine learning or NLP. Moreover for using GraphX, GraphFrames and MLLib, Python is preferred. Python’s visualization libraries complement Pyspark as neither Spark nor Scala have anything comparable.

Code Restoration and safety - Scala is a statically typed language which allows us to find compile time errors. whereas Python is a dynamically typed language. Python language is highly prone to bugs every time you make changes to the existing code. Hence refactoring the code for Scala is easier than refactoring for Python.

Conclusion - Python is slower but very easy to use, while Scala is fastest and moderately easy to use. Scala provides access to the latest features of the Spark, as Apache Spark is written in Scala. Language choice for programming in Apache Spark depends on the features that best fit the project needs, as each one has its own pros and cons. Python is more analytical oriented while Scala is more engineering oriented but both are great languages for building Data Science applications. Overall, Scala would be more beneficial in order to utilize the full potential of Spark. The arcane syntax is worth learning if you really want to do out-of-the-box machine learning over Spark.

https://www.kdnuggets.com/2018/05/apache-spark-python-scala.html

### Applied (7 points total)

Submit your Jupyter Notebook from following along with the code along in [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)
