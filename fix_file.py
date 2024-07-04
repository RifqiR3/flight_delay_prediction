from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# ================= Buka sesi spark =================
spark = SparkSession.builder \
    .appName("FlightDelayPrediction") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()
# ======================================================

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

# ================= Masukkan file csv ===================================
df = spark.read.option("header", "true").option("delimiter", ";").schema(schema).csv("/home/moch1/spark/uas/Jan_2019_ontime_schema_lain.csv")
# =======================================================================

# ================= Verifikasi skema dataset ==================================
df.show(10)
print(f"Total jumlah baris: {df.count()}")
# =============================================================================

# ================= Isi data yang tidak ada dengan 0 ==========================
df = df.na.fill(0)
# =============================================================================

# ================= Ubah ketiga kolom dibawah dalam bentuk numberik =================
categorical_cols = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed") for col in categorical_cols]
# ===================================================================================

# ================= Gabung semua data kolom menjadi satu vektor =================
feature_cols = ["DAY_OF_MONTH", "DAY_OF_WEEK", "DEP_TIME", "ARR_TIME"] + [col + "_indexed" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
# ===============================================================================

# ================= Normalisasi kolom vektor sebelumnya =================
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
# =======================================================================

# ================= Defenisikan kolom untuk label yang akan diprediksi =================
label_col = "DEP_DEL15"
# ======================================================================================

# ================= Inisialisasi algoritma RandomForestClassifier ========================
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=label_col)
# ==============================================================================

# ================= Buat pipepline ==================================
pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])
# ===================================================================

# ================= Definisikan grid parameter untuk hyperparameter tuning guna optimalisasi model =================
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()
# ==================================================================================================================

# ================= Validasi model menggunakan Validasi Silang =================
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol=label_col),
                          numFolds=3)
# ==============================================================================

# ================= Bagi data untuk traning dan testing =================
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
# =======================================================================

# ================= Verifikasi data training dan testing =================
train_data.show(5)
test_data.show(5)
# ========================================================================

# ================= Latih model =================
cvModel = crossval.fit(train_data)
# ===============================================

# ================= Simpan model =================
cvModel.bestModel.write().overwrite().save("/home/moch1/spark/uas/flight_delay_model")
# ================================================

# ================= Buat prediksi untuk tes data =================
predictions = cvModel.transform(test_data)
# ================================================================

# ================= Evaluasi model dan cetak akurasi prediksi =================
evaluator = BinaryClassificationEvaluator(labelCol=label_col)
accuracy = evaluator.evaluate(predictions)
print(f"Test set accuracy = {accuracy}")
# =============================================================================

# ================= Hentikan sesi spark =================
spark.stop()
# =======================================================
