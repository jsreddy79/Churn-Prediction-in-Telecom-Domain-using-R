# Clear complete Work Space
rm(list=ls(all=TRUE))

# Setting the working directory Path
setwd("D:/DA/CSE9099_CPEE Project/data")

#reading data from the csv file
ChurnData = read.csv("Data.csv",header = T,sep = ",")

#To view the column names
names(ChurnData)

#checking for the missinng values
sum(is.na(ChurnData))

#View structure and summary of the data
str(ChurnData)
summary(ChurnData)

#converting target variable to factor
ChurnData$target = as.factor(ChurnData$target)
str(ChurnData$target)

#removing target attribute
S_ChurnData = ChurnData[,-c(18)]

#Standardizing the data Using range method
library(vegan)
S_ChurnData = decostand(S_ChurnData,"range")

#install.packages("car")
library(car)
#install.packages("MASS")
library(MASS)
#install.packages("caret")
library(caret)
#install.packages("mlbench")
library(mlbench)

# Finding the highly correlated columns
highlycorrelated = findCorrelation(S_ChurnData)
highlycorrelated

# Removing the high corelated columns from the data
Final_ChurnData = S_ChurnData[,-c(highlycorrelated)] # Reduced features to 65 from 110

#Adding back the target column to the 'Final_ChurnData' dataframe 
Final_ChurnData$Target = ChurnData$target
names (Final_ChurnData)

# Finding the outliers
Churn_LogReg <- glm(Target~.,data = Final_ChurnData,family = binomial())
influenceIndexPlot(Churn_LogReg,vars = c("cook"),id.n = 5)

#Removing the outliers in the final data
Final_ChurnData<- Final_ChurnData[-c(5821,10433,14739,15054,23022)]

# Splitting the data into train & test with 70% and 30%
rows=seq(1,nrow(Final_ChurnData),1)
set.seed(123)
trainRows=sample(rows,(70*nrow(Final_ChurnData))/100)
train = Final_ChurnData[trainRows,] 
test = Final_ChurnData[-trainRows,]

########################### Logistic Regression ###########################

#Creating Logistic regression on the train dataset
LogReg1 = glm(Target ~ ., family = binomial(),data=train)
summary(LogReg1)

#Predicting the test dataframe using logistic regreesion model created
predictTest = predict(LogReg1, type="response", newdata=test)

#Converting the value into binary
pred_class = factor(ifelse(predictTest> 0.5, 1, 0))
pred_class

#library(caret)
#Confusion Metrics
LR_conf_Matrix=table(test$Target, pred_class)

#Error Metrics
LR_acc = sum(diag(LR_conf_Matrix))/sum(LR_conf_Matrix)*100
LR_prec = LR_conf_Matrix[2,2]/sum(LR_conf_Matrix[,2])*100
LR_recall= LR_conf_Matrix[2,2]/sum(LR_conf_Matrix[2,])*100

#Storing the Error Metrics values into a vector
LogisticReg_Results =c("accuracy"=LR_acc,"precision"=LR_prec, "recall"=LR_recall)
LogisticReg_Results
#################### Features Extraction using AEC #########################

library(h2o)

# Initiate h2o process - can assign ip/port/max_mem_size(ram size)/
# nthreads(no. of processor cores; 2-2core;-1 -all cores available)
localh2o <- h2o.init(ip='localhost', port = 54321, max_mem_size = '1g',nthreads = 1)

#Converting R object to an H2O Object
Final_churn.hex <- as.h2o(localh2o, object = Final_ChurnData, key = "Final_churn.hex")

#To extract features using autoencoder method
aec <- h2o.deeplearning(x = setdiff(colnames(Final_churn.hex), "Target"), 
                        y = "Target", data = Final_churn.hex,
                        autoencoder = T, activation = "RectifierWithDropout",
                        classification = T, hidden = c(30),
                        epochs = 100, l1 = 0.01)

#Converting R object to an H2O Object
train.hex <- as.h2o(localh2o, object = train, key = "train.hex")
test.hex <- as.h2o(localh2o, object = test, key = "test.hex")

################### DeepLearning Model ################################

#DeepLearning model implementation using the AEC features
dl_model = h2o.deeplearning(x = setdiff(colnames(train.hex), "Target"), 
                            y = "Target",
                            data = train.hex, 
                            # activation =  "Tanh", 
                            hidden = c(5, 10, 10),
                            activation = "RectifierWithDropout",
                            input_dropout_ratio = 0.1, 
                            epochs = 100,seed=123)

#Prediction on test data
prediction = h2o.predict(dl_model, newdata = test.hex)

#Convert prediction from h2o object to R object/dataframe
pred = as.data.frame(prediction)

#Confusion Matrix
conf_Matrix=table(test$Target, pred$predict)

#Error Metrics
dl_acc = sum(diag(conf_Matrix))/sum(conf_Matrix)*100
dl_prec = conf_Matrix[2,2]/sum(conf_Matrix[,2])*100
dl_recall= conf_Matrix[2,2]/sum(conf_Matrix[2,])*100

#Storing the Error Metrics values into a vector
DeepLearning_Results =c("accuracy"=dl_acc,"precision"=dl_prec, "recall"=dl_recall)
DeepLearning_Results
#################### Converting AEC extracted features into R ############################

# Converting the AEC extracted features into R dataframe
features = as.data.frame.H2OParsedData(h2o.deepfeatures(Final_churn.hex[,-66], model = aec))

# Adding the Target column to the extracted features
Featured_ChurnData = cbind(features,ChurnData$target)

# Renaming the 'ChurnData$target' column name as 'Target'
names(Featured_ChurnData)[31] = "Target" 

# Split the 'Featured_Churndata' into train and test with 70% & 30%
set.seed(1234)
trainrows = sample(nrow(Featured_ChurnData),0.7 * nrow(Featured_ChurnData))

FCdata_train = Featured_ChurnData[trainrows,]
FCdata_test = Featured_ChurnData[-trainrows,]

###################### Decission Tree - RPART ##############################

library(rpart)

dt_model = rpart(Featured_ChurnData$Target~.,Featured_ChurnData,method = "class")
summary(dt_model)

#Predicting on train data
dt_model_train_pred = predict(dt_model,newdata = FCdata_train,type="class")

#Predicting on train data
dt_model_test_pred = predict(dt_model,newdata = FCdata_test,type="class")

#Confusion matrix for Test data Predictions
DT_conf_Matrix=table(FCdata_test$Target, dt_model_test_pred)

#Error Metrics
DT_acc = sum(diag(DT_conf_Matrix))/sum(DT_conf_Matrix)*100
DT_prec = DT_conf_Matrix[2,2]/sum(DT_conf_Matrix[,2])*100
DT_recall= DT_conf_Matrix[2,2]/sum(DT_conf_Matrix[2,])*100

#Storing the Error Metrics values into a vector
DecissionTree_Results =c("accuracy"=DT_acc,"precision"=DT_prec, "recall"=DT_recall)
DecissionTree_Results
########################## Random Forest  #################################

set.seed(12345)
library(randomForest)

rf_model = randomForest(FCdata_train$Target~.,data=FCdata_train, keep.forest=TRUE, ntree=10) 
summary(rf_model)

# Predict on Train data 
rf_model_train_pred = predict(rf_model,FCdata_train,type="response")

# Predicton Test Data
rf_model_test_pred <-predict(rf_model,FCdata_test,type="response")

#Confusion matrix for Test data Predictions
RF_conf_Matrix=table(FCdata_test$Target, rf_model_test_pred)

#Error Metrics
RF_acc = sum(diag(RF_conf_Matrix))/sum(RF_conf_Matrix)*100
RF_prec = RF_conf_Matrix[2,2]/sum(RF_conf_Matrix[,2])*100
RF_prec= RF_conf_Matrix[2,2]/sum(RF_conf_Matrix[2,])*100

#Storing the Error Metrics values into a vector
RandomForest_Results =c("accuracy"=RF_acc,"precision"=RF_prec, "recall"=RF_prec)
RandomForest_Results
################################## SVM  ##################################
library(e1071)
#SVM on the data which has all columns after removing the highly corelated columns
#svm_model2 <- svm (Target ~.,data = train, kernel = "radial", cost = 10, gamma=0.1) 

#Predict on the test data
#svm_model2_test_pred = predict(svm_model2, test)

#Confusion Metrics on the test data
#svm_conf_Matrix2=table(FCdata_test$Target, svm_model2_test_pred)

#Error Metrics
#svm_acc = sum(diag(svm_conf_Matrix2))/sum(svm_conf_Matrix2)*100
#svm_prec = svm_conf_Matrix2[2,2]/sum(svm_conf_Matrix2[,2])*100
#svm_prec= svm_conf_Matrix2[2,2]/sum(svm_conf_Matrix2[2,])*100

#As the accuracy is low, trying to build the model on the AEC extracted features
#SVM model on the AEC extracted features
svm_model <- svm (Target ~.,data = FCdata_train, kernel = "radial", cost = 10, gamma=0.1) 

#Predicting the model on teh test data
svm_model_test_pred = predict(svm_model, FCdata_test)

#Confusion Metrics for the Test data prediction
svm_conf_Matrix=table(FCdata_test$Target, svm_model_test_pred)

#Error Metrics
svm_acc = sum(diag(svm_conf_Matrix))/sum(svm_conf_Matrix)*100
svm_prec = svm_conf_Matrix[2,2]/sum(svm_conf_Matrix[,2])*100
svm_prec= svm_conf_Matrix[2,2]/sum(svm_conf_Matrix[2,])*100

#Storing the Error Metrics values into a vector
SVM_Results =c("accuracy"=svm_acc,"precision"=svm_prec, "recall"=svm_prec)
SVM_Results
########################### Final Result ####################################

#Final Result displaying the Error Metrics for all the above models
Final_Result = t(data.frame(LogisticReg_Results,
                            DeepLearning_Results,
                            DecissionTree_Results,
                            RandomForest_Results,
                            SVM_Results))
View(Final_Result)
