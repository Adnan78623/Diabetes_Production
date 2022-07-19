from unittest import result
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pandas.plotting import scatter_matrix
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def diabetes_perdiction(glc,bp,bmi,age):

    spark = SparkSession.builder.appName('ml-diabetes').getOrCreate()
    df = spark.read.csv('diabetes.csv', header = True, inferSchema = True)

    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
    numeric_data = df.select(numeric_features).toPandas()

    ## See if we have missing values
    df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

    dataset = df.replace('null', None)\
        .dropna(how='any')

    # Drop unnecessary columns
    dataset = dataset.drop('SkinThickness')
    dataset = dataset.drop('Insulin')
    dataset = dataset.drop('DiabetesPedigreeFunction')
    dataset = dataset.drop('Pregnancies')


    # Assemble all the features with VectorAssembler
    required_features = ['Glucose',
                    'BloodPressure',
                    'BMI',
                    'Age'
                    ]


    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_data = assembler.transform(dataset)

    # Split the data
    (training_data, test_data) = transformed_data.randomSplit([0.8,0.2])

    #Logistic Regression Model
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'Outcome', maxIter=10)
    lrModel = lr.fit(training_data)
    lr_predictions = lrModel.transform(test_data)



    x_test = spark.createDataFrame(
    [
     (float(glc),float(bp),float(bmi),float(age))
    ],
    ["Glucose","BloodPressure","BMI","Age"]
    )      

    x_test.show()
    #type(x_test)

    x_transformed_data = assembler.transform(x_test)
    x_transformed_data.show()

    x_prediction = lrModel.transform(x_transformed_data)
    x_prediction.show()

    x = x_prediction.head()[7]

    if(x == 0):
        result = 2
    else:
        result = 3


    print(result)
    return result

diabetes_perdiction(34,24,12,54)