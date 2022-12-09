
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app_name = 'wine_quality_test'
spark = (
        pyspark.sql.SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0")
        .getOrCreate()
    )
#data = spark.read.format("csv").load("s3://s3mywineproject/test/ValidationDataset.csv", header = True, sep=";")
data = spark.read.format("csv").load("ValidationDataset.csv", header = True, sep=";")
data.printSchema()
data.show()

    
for col_name in data.columns[1:-1] + ['""""quality"""""']:
    data = data.withColumn(col_name, col(col_name).cast('float'))
data = data.withColumnRenamed('""""quality"""""', "label")

features =np.array(data.select(data.columns[1:-1]).collect())
label = np.array(data.select('label').collect())

VectorAssembler = VectorAssembler(inputCols =data.columns[1:-1], outputCol ='features')
df_tr = VectorAssembler.transform(data)
df_tr = df_tr.select(['features','label'])

def to_labeled_point(spark, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    sparkContext=spark.sparkContext
    return sparkContext.parallelize(labeled_points) 

ds = to_labeled_point(spark, features, label)
#RFModel = RandomForestModel.load(spark.sparkContext, "s3://s3mywineproject/model/trainingmodel.model/")
RFModel = RandomForestModel.load(spark.sparkContext, "trainingmodel.model/")
print("model loaded successfully")
predictions = RFModel.predict(ds.map(lambda x: x.features))

labelsAndPredictions = ds.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions_df = labelsAndPredictions.toDF()
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(ds.count())
print('Test Error = ' + str(testErr))

