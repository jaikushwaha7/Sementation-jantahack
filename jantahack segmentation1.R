setwd("D:/Study/Hackathon analytics vidhya/jantahack segmentation 01082020")

traindata<- read.csv("train.csv", header = T, na.strings = c(""," "))
testdata<- read.csv("test.csv", header=T, na.strings = c(""," "))

str(traindata)
anyNA(traindata)
anyNA(testdata)

sapply(traindata,class)
traindata$type <- "train"
testdata$type<- "test"
testdata$Segmentation<- 'A'
data<- rbind(traindata,testdata)
colnames(testdata)
str(data)
cols<- sapply(data, is.character)
cols[T]

library(dplyr)

data<- data %>% mutate_if(is.character, as.factor)
str(data)
summary(data)

data %>% summarise_all(~ sum(is.na(.)))
colSums(is.na(data))
na_count <-sapply(data, function(y) sum(length(which(is.na(y)))))  
na_count

# NA impudatation
#mice
library(mice)
init = mice(data, maxit=0) 
meth = init$method
predM = init$predictorMatrix

meth
imputed = mice(data,  method=meth,predictorMatrix = predM, seed=500)
#imputed = mice(data, m=1, maxit=500, method='cart', seed=500)

imputed <- complete(imputed)

imputed %>% summarise_all(~ sum(is.na(.)))
class(imputed)
summary(imputed)
write.csv(imputed, file = "imputed.csv", row.names = F)

str(imputed)
train1<- imputed[imputed$type=='train',]
dim(train1)
test1<- imputed[imputed$type=='test',]
dim(test1)

prop.table(table(train1$Segmentation))
# target is distributed equally

library(SmartEDA)
ExpData(data=imputed,type=1)
ExpData(data=train1,type=1)
ExpData(data=test1,type=1)

# Desnity plot for numerical data (univariate)
plot1 <- ExpNumViz(imputed,target=NULL,nlim=10,Page=c(2,1),sample=2)
plot1[[1]]
sapply(imputed,class)

# Frequency chart for categorical variable (univariate)
ExpCTable(imputed,Target=NULL,margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T)
summary(imputed)
imputed <- rbind(train1, test1)
imputed <- imputed %>% mutate_all(na_if,"")
summary(imputed)

library(corrplot)
# corplot
data_numeric<- imputed %>% mutate_if(is.factor,as.numeric)
summary(data_numeric)
m<- cor(data_numeric[,-1])
corrplot(m, type = 'upper')

#
# BArplot for categorical data
plot2 <- ExpCatViz(imputed,target=NULL,col ="slateblue4",clim=10,margin=2,Page = c(2,2),sample=4)
plot2[[1]]

# for numeric data
ExpNumStat(imputed,by="GA",gp="Segmentation",Qnt=seq(0,1,0.1),MesofShape=2,Outlier=TRUE,round=2)

# Box plot
plot4 <- ExpNumViz(imputed,target="Segmentation",type=1,nlim=3,fname=NULL,col=c("darkgreen","springgreen3","springgreen1"),Page=c(2,1),sample=3)
plot4[[1]]

imputed %>%
  #filter(Transport == "Car") %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot::corrplot()

# Density plot for numeric data
imputed[-1,] %>%
  select_if(is.numeric) %>%                     
  gather() %>%                            
  ggplot(aes(value)) +                    
  facet_wrap(~ key, scales = "free") +  
  geom_density()    


# Histogram for numeric data
imputed %>%
  select_if(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
#
imputed %>%
  dplyr::select_if(is.factor) %>% 
  tidyr::gather(Measure, value, -Segmentation) %>%                             # Convert to key-value pairs
  ggplot(aes( x = Measure, color=Segmentation)) +                     # Plot the values
  geom_barplot()+
  facet_wrap(~ Measure, scales = "free")    # In separate panels

#splitting data
train1<- imputed[imputed$type=='train',]
dim(train1)
str(train1)
test1<- imputed[imputed$type=='test',]
dim(test1)

library(caTools)
split <- sample.split(train1$Segmentation, SplitRatio = .90)

train2 <-  subset( train1, split==T)
trainval<- subset(train1, split==F)
str(train2)
targettrain2<- train2$Segmentation
targettrainval<- trainval$Segmentation
train2$type<- NULL
trainval$type<- NULL
test1$type<- NULL
targettrain1<-train1$Segmentation
train1$type<- NULL
library(caret)
# One hot encoded data
train1<- data.frame(predict(dummyVars(~.-Segmentation,train1 ), train1))
train1$Segmentation<- targettrain1
str(train1)
train2 <- data.frame(predict(dummyVars(~.-Segmentation,train2 ), train2))
str(train2)
train2$Segmentation<- targettrain2
dim(train2)

trainval <- data.frame(predict(dummyVars(~.-Segmentation,trainval ), trainval))
str(trainval)
trainval$Segmentation<- targettrainval
dim(trainval)

test2 <- data.frame(predict(dummyVars(~.-Segmentation,test1 ), test1))
str(test2)
dim(test2)
test1 <- data.frame(predict(dummyVars(~.-Segmentation,test1 ), test1))
str(test2)
dim(test2)


write.csv(train2,"train2.csv", row.names = F )
write.csv(trainval,"trainval.csv", row.names = F )
write.csv(test2,"test2.csv", row.names = F )
write.csv(train1,'train1.csv', row.names = F)
# So, multi collinearity does not affect the Naive Bayes
# MOdel 1 Naive Bayes
train_control <- trainControl(
  method = "cv", 
  number = 3   )
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)  )

library(caret)
str(train2)
nb.m1 <- caret::train(
  Segmentation ~ ., data = train2[,-1],
  method = "nb",
  trControl = train_control,
  tuneGrid = search_grid,
  preProc = c("BoxCox", "center", "scale", "pca")
)
y#warnings()
# top 5 modesl
nb.m1$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# plot search grid results
plot(nb.m1)

confusionMatrix(nb.m1)  # train accuracy # .4823
# Test prediction
data_pred <- predict(nb.m1,test2[,-1],type="raw")
head(data_pred)
table(data_pred)
#confusionMatrix(data_pred,cartest[,9], positive = "a", mode="everything")  #  test accuracy 83.78
submission<- read.csv('sample_submission.csv', header = T)
head(submission)
#names(submission)<- 'ID'
submission$Segmentation<- data_pred
write.csv(submission,"submissionnb1.csv", row.names = F)
###################################################################################################

####   KNN
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3,classProbs=FALSE,summaryFunction = multiClassSummary)
knnFit <- caret::train(Segmentation ~ ., data = train1[,-1], method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
data_pred <- predict(knnFit,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionknn1.csv", row.names = F)

###################################################################################################

#### LOgistic Regressin

# Logistic Regression 1.1 without smote
trControl <- trainControl(method  = "cv", number  = 10,verboseIter = FALSE,
                          summaryFunction = multiClassSummary)
#trControl$sampling<- 'smote'
fit_glm = caret::train(
  Segmentation ~ .,
  data = train1[,-1],
  method = "multinom",
  trControl = trControl,
  preProcess = c("center", "scale"),
  trace = FALSE
)
data_pred <- predict(fit_glm,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionglm1.csv", row.names = F)

#####
# Logistic Regression 1.1 with smote
trControl <- trainControl(method  = "cv", number  = 10,verboseIter = FALSE,
                          summaryFunction = multiClassSummary)
trControl$sampling<- 'smote'
fit_glm = caret::train(
  Segmentation ~ .,
  data = train1[,-1],
  method = "multinom",
  trControl = trControl,
  preProcess = c("center", "scale"),
  trace = FALSE
)
data_pred <- predict(fit_glm,test2[,-1],type="raw")
data_pred
submission$Segmentation<- data_pred
write.csv(submission,"submissionglm2.csv", row.names = F)

###############################################################################

#Bagging 1.1
library(rpart)
cntrl <- trainControl(method = "cv", number = 10)
control = rpart.control(minsplit = 2, cp = 0)
fit_bag<- caret::train(Segmentation ~ ., data = train1[,-1], method = "treebag",nbagg= 200, trControl = cntrl)
data_pred <- predict(fit_bag,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionbagging1.csv", row.names = F)





##############################################################################
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(7)
mtry <- sqrt(ncol(train1))
rf_random <- caret::train(Segmentation~., data=train1[,-1], method="rf",
                          metric="Accuracy", tuneLength=15, trControl=control)
data_pred <- predict(rf_random,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionrf1.csv", row.names = F)

################################################################################

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(7)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- caret::train(Segmentation~., data=train1[,-1], method="rf", metric="Accuracy",
                              tuneGrid=tunegrid, trControl=control)
data_pred <- predict(rf_gridsearch,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionrf2.csv", row.names = F)
##################################################################################
# Boosting  1.1
Control <- trainControl(  method = "repeatedcv",  number = 10, repeats = 3)
set.seed(825)
gbmFit1 <- caret::train(Segmentation ~ ., data = train1[,-1], 
                        method = "gbm", 
                        trControl = Control,
                        verbose = FALSE)

data_pred <- predict(gbmFit1,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissiongbm1.csv", row.names = F)

###################################################################################

####   XGBOOST
# Specify the model and the parameters to tune (parnsip)
library(xgboost)
library(tidyverse)
library(tidymodels)
model <-
  boost_tree(tree_depth = tune(), mtry = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

# Specify the resampling method (rsample)
splits <- vfold_cv(train1[,-1], v = 2)

# Specify the metrics to optimize (yardstick)
metrics <- metric_set(roc_auc)

# Specify the parameters grid (or you can use dials to automate your grid search)
grid <- expand_grid(tree_depth = c(4, 6, 8, 10),
                    mtry = c(2, 10, 50)) # You can add others

# Run each model (tune)
tuned <- tune_grid(formula = Segmentation ~ .,
                   model = model,
                   resamples = splits,
                   grid = grid,
                   metrics = metrics,
                   control = control_grid(verbose = TRUE))
data_pred <- predict(tuned,test2[,-1],type="raw")
submission$Segmentation<- data_pred
write.csv(submission,"submissionxgb1.csv", row.names = F)

######################################################################################
# Update model
tuned_model <- 
  model %>% 
  finalize_model(select_best(tuned)) %>% 
  fit(Segmentation ~ ., data = train1[,-1])

data_pred <- predict(tuned_model,test2[,-1],type="raw")
head(data_pred)
colnames(data_pred)<-c('A','B','C','D')
pred<-colnames(data_pred)[apply(data_pred,1,function(x) which(x==max(x)))]
pred
submission$Segmentation<- pred
write.csv(submission,"submissionxgb2.csv", row.names = F)

#########################################################

###### neural net
library(neuralnet)
n <- names(train1[,-1])
f <- as.formula(paste("Segmentation~", paste(n[!n %in% "Segmentation"], collapse = " + ")))
f
nn <- neuralnet(f,data=train1[,-1],hidden=10,err.fct="sse",
                linear.output=FALSE,algorithm="backprop",learningrate=0.35)
nn
data_pred <- compute(nn,test2[,-1])
head(data_pred$net.result)
dim(data_pred$net.result)
colnames(data_pred$net.result)<-c('A','B','C','D')
pred<-colnames(data_pred$net.result)[apply(data_pred$net.result,1,function(x) which(x==max(x)))]
pred
table(pred)
submission$Segmentation<- pred
write.csv(submission,"submissionnn1.csv", row.names = F)
##########################################################################

tune.grid.neuralnet <- expand.grid(
  layer1 = 10,
  layer2 = 10,
  layer3 = 10
)

nn2 <- caret::train(
  f,
  data = train1[,-1],
  method = "neuralnet",
  linear.output = TRUE, 
  tuneGrid = tune.grid.neuralnet, # cannot pass parameter hidden directly!!
  metric = "RMSE",
  trControl = trainControl(method = "Accuracy", seeds = seed)
)


##############################################################################

### Neural net try 3
# Scale data

#splitting data
train1<- imputed[imputed$type=='train',]
dim(train1)
str(train1)
train1$type<- NULL
test1<- imputed[imputed$type=='test',]
dim(test1)
str(test1)

scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
targettrain1 <- train1$Segmentation
train1 <- train1 %>% mutate_if(is.factor, as.numeric)
str(train1)
train1$Segmentation<- targettrain1
str(test1)
test1 <- test1 %>% mutate_if(is.factor, as.numeric)
str(test1)
train1[, 2:10] <- data.frame(lapply(train1[, 2:10], scl))
str(train1)
test1[, 2:10] <- data.frame(lapply(test1[, 2:10], scl))
str(test1)
summary(test1)
library(nnet)
# Encode as a one hot vector multilabel data
train1 <- cbind(train1[, 1:10], class.ind(as.factor(train1$Segmentation)))
str(train1)

# Set up formula
n <- names(train1[,-1])
f <- as.formula(paste("Segmentation~", paste(n[!n %in% c("A","B","C","D","Segmentation")], collapse = " + ")))
f

nn <- neuralnet(f,
                data = train1[,-1],
                hidden = c(13, 10, 3),
                #act.fct = "logistic",
                linear.output = FALSE,
                lifesign = "minimal")

train_control <- trainControl(method="cv", number=10)
# train the model
grid <- expand.grid(layer1 = c(1,5,6))
model <- train(Segmentation~., data=train1[,-1], trControl=train_control, method="nnet", 
               algorithm = "logistic", learningrate = 0.25,act.fct = 'tanh',
               tuneGrid = nnetGrid ,threshold = 0.1
               )
data_pred <- predict(nn,test1[,-1])
head(data_pred)
dim(data_pred)
table(data_pred)
colnames(data_pred$net.result)<-c('A','B','C','D')
pred<-colnames(data_pred$net.result)[apply(data_pred$net.result,1,function(x) which(x==max(x)))]
pred
table(pred)
submission$Segmentation<- pred
write.csv(submission,"submissionnn2.csv", row.names = F)

# summarize results
testmat<- matrix(test1[,-1])
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = multiClassSummary)

nnetGrid <-  expand.grid(# layers= data.frame(layer1 = 2:6, layer2 = 2:6, layer3 = 0),
                          size = seq(from = 1, to = 10, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

nnetFit <- train(Segmentation ~ ., 
                 data = train1[,-1],
                 method = "nnet"
                 )



data_pred <- neuralnet::compute(nn,test1[,-1])$net.result
head(data_pred$net.result)
dim(data_pred$net.result)
colnames(data_pred$net.result)<-c('A','B','C','D')
pred<-colnames(data_pred$net.result)[apply(data_pred$net.result,1,function(x) which(x==max(x)))]
pred
table(pred)
submission$Segmentation<- pred
write.csv(submission,"submissionnn2.csv", row.names = F)
train1<- train1[,-c(11:14)]
str(train1)
write.csv(train1, 'train3.csv', row.names = F)
write.csv(test1,'test3.csv', row.names = F)
