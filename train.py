
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
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app_name = 'wine_quality_train'
spark = (
        pyspark.sql.SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0")
        .getOrCreate()
    )

df = spark.read.format("csv").load("s3://s3mywineproject/train/TrainingDataset.csv" , header = True ,sep =";")
df.show(truncate=False)
for col_name in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")

features =np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('label').collect())

VectorAssembler = VectorAssembler(inputCols = df.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(df)
df_tr = df_tr.select(['features','label'])

def to_labeled_point(spark, features, labels):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    sparkContext=spark.sparkContext
    return sparkContext.parallelize(labeled_points) 

ds = to_labeled_point(spark, features, label)
training, test = ds.randomSplit([0.7, 0.3], seed =11)
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

#predictions
predictions = RFmodel.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions_df = labelsAndPredictions.toDF()
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show(truncate=False)
labelpred_df = labelpred.toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("The f1 score is : ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy : " , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())    
print('Test Error is : ' + str(testErr))
RFmodel.save(spark.sparkContext, 's3://s3mywineproject/model/trainingmodel.model')



