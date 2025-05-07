from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("MovieDataProcessing").getOrCreate()

# Load raw data
raw_data = spark.read.csv("data/raw/ratings.csv", header=True, inferSchema=True)

# Clean and transform data
processed_data = raw_data.groupBy("userId", "movieId").agg({"rating": "mean"})

# Save processed data
processed_data.write.parquet("data/processed/processed_ratings.parquet")

spark.stop()