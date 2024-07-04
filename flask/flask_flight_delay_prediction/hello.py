from flask import Flask, request, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import hour, minute
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
import pandas as pd

app = Flask(__name__)

# Create a Spark session
spark = SparkSession.builder.appName("FlightDelayPredictionWeb").getOrCreate()

# Load the trained model
model = PipelineModel.load("/home/moch1/spark/uas/flight_delay_model")

# Define the schema for the input data
schema = StructType([
    StructField("DAY_OF_MONTH", IntegerType(), True),
    StructField("DAY_OF_WEEK", IntegerType(), True),
    StructField("OP_UNIQUE_CARRIER", StringType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("DEP_TIME", IntegerType(), True),
    StructField("DEP_DEL15", DoubleType(), True),
    StructField("ARR_TIME", IntegerType(), True),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = {
        "DAY_OF_MONTH": int(request.form['DAY_OF_MONTH']),
        "DAY_OF_WEEK": int(request.form['DAY_OF_WEEK']),
        "OP_UNIQUE_CARRIER": request.form['OP_UNIQUE_CARRIER'],
        "ORIGIN": request.form['ORIGIN'],
        "DEST": request.form['DEST'],
        "DEP_TIME": int(request.form['DEP_TIME']),
        "ARR_TIME": int(request.form['ARR_TIME']),
    }
    
    # Create a DataFrame from the input data
    input_df = spark.createDataFrame([input_data], schema=schema)
    
    # Make predictions
    predictions = model.transform(input_df)
    prediction = predictions.select("prediction").collect()[0][0]
    
    # Render the prediction result
    return render_template('index.html', prediction_text=f'Flight delay prediction: {"Delayed" if prediction == 1.0 else "Not Delayed"}')

if __name__ == "__main__":
    app.run(debug=True)
