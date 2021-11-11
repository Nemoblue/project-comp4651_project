# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Project Description
# MAGIC 
# MAGIC In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.
# MAGIC 
# MAGIC This Project aims to predict players' finishing placement based on their final stats in a PUBG game. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 player per group.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Explore Data
# MAGIC From the .csv file import dataframe. The delimiter is comma and the first row is the header.

# COMMAND ----------

from pyspark.sql.types import *
# File location and type
file_location = "/FileStore/tables/train_V2-1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# To increase the accuracy, we set all numerical values as doubleType
customSchema = StructType([ \
    StructField("Id", StringType(), True),
    StructField("groupId", StringType(), True),
    StructField("matchId", StringType(), True),
    StructField("assists", DoubleType(), True), \
    StructField("boosts", DoubleType(), True), \
    StructField("damageDealt", DoubleType(), True), \
    StructField("DBNOs", DoubleType(), True), \
    StructField("headshotKills", DoubleType(), True), \
    StructField("heals", DoubleType(), True), \
    StructField("killPlace", DoubleType(), True), \
    StructField("killPoints", StringType(), True),
    StructField("kills", DoubleType(), True), \
    StructField("killStreaks", DoubleType(), True), \
    StructField("longestKill", DoubleType(), True),
    StructField("matchDuration", DoubleType(), True),
    StructField("matchType", StringType(), True),
    StructField("maxPlace", DoubleType(), True),
    StructField("numGroups", DoubleType(), True),
    StructField("rankPoints", DoubleType(), True),
    StructField("revives", DoubleType(), True), \
    StructField("rideDistance", DoubleType(), True), \
    StructField("roadKills", DoubleType(), True), \
    StructField("swimDistance", DoubleType(), True), \
    StructField("teamKills", DoubleType(), True),
    StructField("vehicleDestroys", DoubleType(), True), \
    StructField("walkDistance", DoubleType(), True), \
    StructField("weaponsAcquired", DoubleType(), True),
    StructField("winPoints", DoubleType(), True),
    StructField("winPlacePerc", DoubleType(), True)])

# The applied options are for CSV files. For other file types, these will be ignored.
originDF = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location, schema = customSchema)

display(originDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the type of all data. Notice that some of them are double and string instead of int.

# COMMAND ----------

print originDF.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Preliminary Analyses
# MAGIC Then we will do some original analyses of the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC **Schema Definition**
# MAGIC 
# MAGIC Here's the schema definition:
# MAGIC 
# MAGIC  - DBNOs - Number of enemy players knocked.
# MAGIC  - assists - Number of enemy players this player damaged that were killed by teammates.
# MAGIC  - boosts - Number of boost items used.
# MAGIC  - damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# MAGIC  - headshotKills - Number of enemy players killed with headshots.
# MAGIC  - heals - Number of healing items used.
# MAGIC  - Id - Player’s Id
# MAGIC  - killPlace - Ranking in match of number of enemy players killed.
# MAGIC  - killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# MAGIC  - killStreaks - Max number of enemy players killed in a short amount of time.
# MAGIC  - kills - Number of enemy players killed.
# MAGIC  - longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# MAGIC  - matchDuration - Duration of match in seconds.
# MAGIC  - matchId - ID to identify match. There are no matches that are in both the training and testing set.
# MAGIC  - matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# MAGIC  - rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# MAGIC  - revives - Number of times this player revived teammates.
# MAGIC  - rideDistance - Total distance traveled in vehicles measured in meters.
# MAGIC  - roadKills - Number of kills while in a vehicle.
# MAGIC  - swimDistance - Total distance traveled by swimming measured in meters.
# MAGIC  - teamKills - Number of times this player killed a teammate.
# MAGIC  - vehicleDestroys - Number of vehicles destroyed.
# MAGIC  - walkDistance - Total distance traveled on foot measured in meters.
# MAGIC  - weaponsAcquired - Number of weapons picked up.
# MAGIC  - winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# MAGIC  - groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# MAGIC  - numGroups - Number of groups we have data for in the match.
# MAGIC  - maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# MAGIC  - winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Notice that some of the schema is not related to the winPlace, so we remove them right away.** 
# MAGIC 
# MAGIC  - Id - Player’s Id
# MAGIC  - groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# MAGIC  - matchId - ID to identify match. There are no matches that are in both the training and testing set.
# MAGIC  
# MAGIC **To simplify the algorithm, we decide to remove all statistics related to ELO ranking**
# MAGIC 
# MAGIC  - killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# MAGIC  - rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# MAGIC  - winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# MAGIC  
# MAGIC **Since there are some statistics may confuse the results due to complex match cases, we remove them instead of dealing with them**
# MAGIC  - longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# MAGIC  - matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.

# COMMAND ----------

trainDF = originDF.drop('Id', 'groupId', 'matchId', 'killPoints', 'rankPoints', 'winPoints', 'longestKill', 'matchType')
display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **Then we register our DataFrame as an SQL table named "project.dataset", and do some basic statistical analyses of all the columns.**

# COMMAND ----------

sqlContext.sql("DROP TABLE IF EXISTS project_dataset")
dbutils.fs.rm("dbfs:/user/hive/warehouse/project_dataset", True)
sqlContext.registerDataFrameAsTable(trainDF, "project_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we need to examine some of statistics that are unrelated to the winplace percentage.**
# MAGIC **We choose walkDistance, killPlace, matchDuration, maxPlace, numGroups, teamKills as our example.**

# COMMAND ----------

# MAGIC %sql
# MAGIC select walkDistance, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select killPlace, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select matchDuration, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select maxPlace, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select numGroups, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC select teamKills, winPlacePerc from project_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC **As a conclusion, we find that matchDuration, maxPlace, numGroups, teamKills have little effect on our target, but walkDistance and killPlace has a certain effect. So we need to drop these three columns to decrease the number of calculations**

# COMMAND ----------

trainDF = trainDF.drop('matchDuration', 'maxPlace', 'numGroups', 'teamKills')

## Drop rows with null value
trainDF = trainDF.na.drop()

sqlContext.sql("DROP TABLE IF EXISTS project_dataset")
dbutils.fs.rm("dbfs:/user/hive/warehouse/project_dataset", True)
sqlContext.registerDataFrameAsTable(trainDF, "project_dataset")

display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we can check whether there is a null value in the dataset. If there is one null value, we have to drop this row.**
# MAGIC  - Since we have handled the null values in some rows before, it should have no outputs.

# COMMAND ----------

# MAGIC %sql
# MAGIC select winPlacePerc from project_dataset
# MAGIC where winPlacePerc is null

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 3: Data Preparation
# MAGIC 
# MAGIC Then we need to prepare the data for machine learning. 
# MAGIC 
# MAGIC - Convert the `project_dataset` SQL table into a DataFrame.
# MAGIC - Set the vectorizer's input columns to a list of the sixteen columns of the input DataFrame.
# MAGIC - Set the vectorizer's output column name to `"features"`.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

datasetDF = sqlContext.table("project_dataset")

vectorizer = VectorAssembler()
vectorizer.setInputCols(["assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace", "kills", "killStreaks", "revives", "rideDistance", "roadKills", "swimDistance", "vehicleDestroys", "walkDistance", "weaponsAcquired"])
vectorizer.setOutputCol("features")

# COMMAND ----------

# MAGIC %md
# MAGIC **In order to see how well the model is, we need to split the dataset into training set and test set. Then we cache them.**

# COMMAND ----------

(split15DF, split85DF) = datasetDF.randomSplit([0.15, 0.85], seed = "20583761")
testSetDF = split15DF.cache()
trainingSetDF = split85DF.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC **Before we apply any of the algorithm, we first generate linear analyses of the dataset to see some preliminary results**

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

lr = LinearRegression()
lr.setPredictionCol("Prediction_WP")\
  .setLabelCol("winPlacePerc")\
  .setMaxIter(100)\
  .setRegParam(0.15)

lrPipeline = Pipeline()
lrPipeline.setStages([vectorizer, lr])
lrModel = lrPipeline.fit(trainingSetDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **Then we can draw the equation of the linear regrassion to see a preliminary results.**

# COMMAND ----------

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "winPlacePerc"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

# Now let's sort the coefficients from greatest absolute weight most to the least absolute weight
coefficents.sort(key=lambda tup: abs(tup[0]), reverse=True)

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)

# COMMAND ----------

# MAGIC %md
# MAGIC **Next we apply the Linear Regression model to the 15% of the data that we split from the input dataset. The output of the model will be a predicted win place named "Prediction_WP". Then we calculate its rmse and \\(r^2\\)**

# COMMAND ----------

resultsDF = lrModel.transform(testSetDF).select("assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace", "kills", "killStreaks", "revives", "rideDistance", "roadKills", "swimDistance", "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPlacePerc", "Prediction_WP")
display(resultsDF)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator(predictionCol="Prediction_WP", labelCol="winPlacePerc", metricName="rmse")
rmse = regEval.evaluate(resultsDF)
r2 = regEval.evaluate(resultsDF, {regEval.metricName: "r2"})

print("Root Mean Squared Error: %.2f" % rmse)
print("r2: {0:.2f}".format(r2))

# COMMAND ----------

# MAGIC %md
# MAGIC **Then we can construct the pie chart of RMSE to see whether a linear regression is perfect.**

# COMMAND ----------

sqlContext.sql("DROP TABLE IF EXISTS Project_RMSE_Evaluation")
dbutils.fs.rm("dbfs:/user/hive/warehouse/Project_RMSE_Evaluation", True)

# Next we calculate the residual error and divide it by the RMSE
resultsDF.selectExpr("winPlacePerc", "Prediction_WP", "winPlacePerc - Prediction_WP Residual_Error", "(winPlacePerc - Prediction_WP) / {} Within_RSME".format(rmse)).registerTempTable("Project_RMSE_Evaluation")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT case 
# MAGIC        when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1
# MAGIC        when Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 
# MAGIC        else 3
# MAGIC        end RSME_Multiple, COUNT(*) AS count
# MAGIC FROM Project_RMSE_Evaluation
# MAGIC GROUP BY RSME_Multiple

# COMMAND ----------

# MAGIC %md
# MAGIC From the pie chart, we can see that 69% of our test data predictions are within 1 RMSE of the actual values, and 96% (69% + 27%) of our test data predictions are within 2 RMSE. So the model is pretty decent. Then we can use tuning to improve it.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 4: Evaluation with Tuning
# MAGIC 
# MAGIC To improve the results, we can directly use decision tree and random forest to get the improvement. But firstly we can use the decision tree to see its accuracy. Then we use the random forest to see whether there can be more accuracy.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import DecisionTreeRegressor

# Create a DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.setPredictionCol("Prediction_WP").setFeaturesCol("features").setLabelCol("winPlacePerc").setMaxBins(100)

# Create a Pipeline
dtPipeline = Pipeline()

# Set the stages of the Pipeline
dtPipeline.setStages([vectorizer, dt])

# Reuse the CrossValidator
crossval = CrossValidator(estimator=dtPipeline, evaluator=regEval, numFolds=3)

# Create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [14, 15])
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Find and return the best model
dtModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC **We can evaluate the model by calculate DT model's RMSE and \\(r^2\\) values.**

# COMMAND ----------

resultsDF = dtModel.transform(testSetDF).select("assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace", "kills", "killStreaks", "revives", "rideDistance", "roadKills", "swimDistance", "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPlacePerc", "Prediction_WP")
rmseDT = regEval.evaluate(resultsDF)
r2DT = regEval.evaluate(resultsDF, {regEval.metricName: "r2"})

print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("DT r2: {0:.2f}".format(r2DT))

# COMMAND ----------

# MAGIC %md
# MAGIC The line below will pull the Decision Tree model from the Pipeline as display it as an if-then-else string.

# COMMAND ----------

print dtModel.stages[-1]._java_obj.toDebugString()

# COMMAND ----------

# MAGIC %md
# MAGIC **Notice that the depth of dt we estimated is 14 or 15 which is slightly small. So we have to adopt random tree to see whether it can be improved. And instead of guessing the parameters, we use Model Selection or Hyperparameter Tuning to select the best model**
# MAGIC 
# MAGIC The parameters for the method list below:
# MAGIC 
# MAGIC  - Set the name of the prediction column to "Prediction_WP"
# MAGIC  - Set the name of the label column to "winPlacePerc"
# MAGIC  - Set the name of the features column to "features"
# MAGIC  - Set the random number generator seed to 20583761
# MAGIC  - Set the maximum depth to 10
# MAGIC  - Set the number of trees to 24

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Create a RandomForestRegressor
rf = RandomForestRegressor()
rf.setPredictionCol("Prediction_WP").setLabelCol("winPlacePerc").setFeaturesCol("features").setSeed(20583761).setMaxDepth(10).setNumTrees(24)

# Create a Pipeline and set the stages of the Pipeline
rfPipeline = Pipeline()
rfPipeline.setStages([vectorizer, rf])

# Reuse the CrossValidator
crossval.setEstimator(rfPipeline)

# Tune over rf.maxBins parameter on the values 50 and 100, create a paramter grid using the ParamGridBuilder
paramGrid = ParamGridBuilder().addGrid(rf.maxBins, [50, 100]).build()

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Find and return the best model
rfModel = crossval.fit(trainingSetDF).bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can get the tuned RandomForestRegressor model's RMSE and \\(r^2\\) values and compare them to our tuned DecisionTreeRegressor models.

# COMMAND ----------

resultsDF = rfModel.transform(testSetDF).select("assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace", "kills", "killStreaks", "revives", "rideDistance", "roadKills", "swimDistance", "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPlacePerc", "Prediction_WP")
rmseRF = regEval.evaluate(resultsDF)
r2RF = regEval.evaluate(resultsDF, {regEval.metricName: "r2"})

print("LR Root Mean Squared Error: {0:.2f}".format(rmse))
print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("RF Root Mean Squared Error: {0:.2f}".format(rmseRF))
print("LR r2: {0:.2f}".format(r2))
print("DT r2: {0:.2f}".format(r2DT))
print("RF r2: {0:.2f}".format(r2RF))

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately, the performance of random tree is not as good as decision tree. But we still get an optimal result with relatively low RMSE and high \\(r^2\\). Then we pull the Random Forest model from the Pipeline as an if-then-else string.

# COMMAND ----------

print rfModel.stages[-1]._java_obj.toDebugString()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Conclusion
# MAGIC 
# MAGIC Till now, we have generated three kinds of models to evaluate the winPlacePerc, which represents of win place percentage. For the last two models, we can find that feature 6(killPlace) and feature 14(walkDistance) has a huge effect on our target. But considering the fact that the person who gets the champion is more likely to have a high killPlace and walk a long distance, we can look at the linear regression expression to get some ideas:
# MAGIC 
# MAGIC `Linear Regression Equation: y = 0.413646647358 + (0.0230870694198 * boosts) + (0.0179682771692 * assists) + ...`
# MAGIC 
# MAGIC From the equation, we notice that boosts, assists and weaponsAcquire have a high coefficient. We may get some conclusions saying that more boosts you throw, more assists you get, and more weapons you pick, you may have more chance to get the champion and eat the chicken!

# COMMAND ----------

# MAGIC %md
# MAGIC In the end, we apply our best model of decision tree to generate the results of the testSet given by the websites.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/train_V2-2.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
testDF = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(testDF)

# COMMAND ----------

# Apply the best model
resultsDF = dtModel.transform(testDF).select("Id", "Prediction_WP")
display(resultsDF)
