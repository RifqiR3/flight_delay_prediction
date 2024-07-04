from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, hour, minute
from pyspark.ml import PipelineModel

# ================= Buka sesi spark =================
spark = SparkSession.builder \
    .appName("FlightDelayPrediction") \
    .getOrCreate()
# ======================================================

# ================= Masukkan model =================================
model_path = "/home/moch1/spark/uas/flight_delay_model"
loaded_model = PipelineModel.load(model_path)
# ==================================================================

# ================= Defenisikan skema sesuai dengan dataset =================
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
# ============================================================================

# ================= Masukkan file csv yang akan diprediksi ===================================
new_data = spark.read.option("header", "true").option("delimiter", ";").schema(schema).csv("/home/moch1/spark/uas/Feb_2019_ontime_schema_lain.csv")
# ============================================================================================

# ================= Buat prediksi untuk dataset yang baru =================
predictions = loaded_model.transform(new_data)
# =========================================================================

# ================= Lihat hasil prediksi untuk 100 data pertama =====================================================================================
predictions.select("DAY_OF_MONTH", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME", "ARR_TIME", "DEP_DEL15", "prediction") \
    .show(100, truncate=False)
# ===================================================================================================================================================

# ================= Lihat jumlah prediksi yang benar dan tingkat akurasi modelnya =======================
correct_predictions = predictions.filter(predictions["DEP_DEL15"] == predictions["prediction"]).count()
total_predictions = predictions.count()
accuracy = correct_predictions / total_predictions
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {accuracy}")
# =======================================================================================================


# ================= Hentikan sesi spark =================
spark.stop()
# =======================================================