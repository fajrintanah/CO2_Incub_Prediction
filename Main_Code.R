library(tidyverse) # plotting and manipulation
library(grid) # combining plots
library(gridExtra) # combining plots
library(ggpubr) # combining plots
library(patchwork) # combining plots
library(ggfortify) # nice extension for ggplot
library(mgcv) #fitting gam models
library(GGally) # displaying pairs panel
library(FactoMineR)#PCANguyen
library(factoextra) #PCANguyen
library(survminer)
library(multcompView)
library(Hmisc)
library(nlme)
library(lme4)
library(vegan)
library(caTools) # split dataset
library(readxl)
library(randomForest)
library(e1071)
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(pdp)          # model visualization
library(lime)         # model visualization
library(neuralnet)
library(rpart)     #rpart for computing decision tree models
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(broom)
library(ranger) 	#efficient RF
library(relaimpo) #extract relative importance
library(rminer)
library(NeuralNetTools)
library(vip)
library(tidymodels)
library(earth)
library(iml)
library(car)


library(readxl)
Data_S <- read_excel("D:/Publikasi/Pak Heru B Pulunggono/7 Simulating and Modeling CO2 flux emited from decomposed oil palm root - HBP, SF, DN, SA, LLN, MZF/Data/Data_S.xlsx", 
    sheet = "R_fix")
View(Data_S)



str(Data_S)
		tibble [60 x 6] (S3: tbl_df/tbl/data.frame)
		 $ Water_Content  : num [1:60] 15 15 15 15 15 15 15 15 15 15 ...
		 $ Incubation_Time: num [1:60] 0 0 0 0 0 1 1 1 1 1 ...
		 $ CO2_Flux       : num [1:60] 0.015 0.0244 0.0245 0.0272 0.0102 ...
		 $ Org_C          : num [1:60] 56 55.7 55.7 55.4 56 ...
		 $ N_Total        : num [1:60] 0.456 0.402 0.322 0.456 0.483 ...
		 $ CN             : num [1:60] 123 138 173 121 116 ...

## Data_R_YNS_baru1<-Data_R_YNS[-c(62:64),]
## Data_R_YNS_baru<-Data_R_YNS_baru1[-c(1),]

summary(Data_S) 
		 Water_Content   Incubation_Time    CO2_Flux            Org_C          N_Total      
		 Min.   : 15.0   Min.   :0.000   Min.   :0.002966   Min.   :53.83   Min.   :0.3219  
		 1st Qu.: 15.0   1st Qu.:2.000   1st Qu.:0.014167   1st Qu.:54.41   1st Qu.:0.4289  
		 Median :125.0   Median :2.000   Median :0.034049   Median :54.77   Median :0.4557  
		 Mean   :171.7   Mean   :2.167   Mean   :0.042714   Mean   :54.79   Mean   :0.4562  
		 3rd Qu.:300.0   3rd Qu.:3.000   3rd Qu.:0.062096   3rd Qu.:55.04   3rd Qu.:0.4784  
		 Max.   :450.0   Max.   :3.000   Max.   :0.143093   Max.   :55.99   Max.   :0.5358  
			   CN       
		 Min.   :102.3  
		 1st Qu.:113.5  
		 Median :120.2  
		 Mean   :121.1  
		 3rd Qu.:126.9  
		 Max.   :173.1 


maxs <- apply(Data_S, 2, max) 
mins <- apply(Data_S, 2, min)
scaled_syva <- as.data.frame(scale(Data_S, center = mins, 
                              scale = maxs - mins))	



set.seed(42)

sampleSplit_flux <- sample.split(Y=scaled_syva$CO2_Flux, SplitRatio=0.7)
trainSet_flux <- subset(x=scaled_syva, sampleSplit_flux==TRUE)
testSet_flux <- subset(x=scaled_syva, sampleSplit_flux==FALSE)

x_CO2_Flux_train = subset(trainSet_flux, select = -CO2_Flux) %>% as.matrix() 
y_CO2_Flux_train = trainSet_flux$CO2_Flux

x_CO2_Flux_test = subset(testSet_flux, select = -CO2_Flux) %>% as.matrix() 
y_CO2_Flux_test = testSet_flux$CO2_Flux



set.seed = 42
LM_flux <- train(
  x = x_CO2_Flux_train,
  y = y_CO2_Flux_train,
  method = "lm",
  family = "gaussian",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10)
)

summary(LM_flux)

Call:
lm(formula = .outcome ~ ., data = dat, family = "gaussian")

Residuals:
     Min       1Q   Median       3Q      Max 
-0.55519 -0.16516 -0.02446  0.10970  0.59423 

Coefficients:
                Estimate Std. Error t value Pr(>|t|)   
(Intercept)       0.7696     1.1043   0.697  0.49036   
Water_Content    -0.1138     0.1345  -0.846  0.40312   
Incubation_Time  -0.6371     0.2091  -3.046  0.00432 **
Org_C            -0.6625     0.2848  -2.326  0.02574 * 
N_Total           0.5085     1.2281   0.414  0.68132   
CN                0.0695     1.3858   0.050  0.96028   
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.2544 on 36 degrees of freedom
Multiple R-squared:  0.2405,	Adjusted R-squared:  0.135 
F-statistic:  2.28 on 5 and 36 DF,  p-value: 0.06718

preds_LM_flux <- predict(LM_flux, testSet_flux)

modelEval_LM_flux <- cbind(testSet_flux$CO2_Flux, preds_LM_flux)
colnames(modelEval_LM_flux) <- c('Actual', 'Predicted')
modelEval_LM_flux <- as.data.frame(modelEval_LM_flux)

mse_LM_flux <- mean((modelEval_LM_flux$Actual - modelEval_LM_flux$Predicted)^2)
rmse_LM_flux <- sqrt(mse_LM_flux)
rmse_LM_flux
0.2211144




hyper_grid_mars <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
  )

set.seed(42)

# cross validated model
mars_flux1 <- train(
  x = x_CO2_Flux_train,
  y = y_CO2_Flux_train,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneGrid = hyper_grid_mars
)

# best model
mars_flux1$bestTune
   nprune degree
22     12      3

ggplot(mars_flux1)


preds_MARS_flux <- predict(mars_flux1, testSet_flux)

modelEval_MARS_flux <- cbind(testSet_flux$CO2_Flux, preds_MARS_flux)
colnames(modelEval_MARS_flux) <- c('Actual', 'Predicted')
modelEval_MARS_flux <- as.data.frame(modelEval_MARS_flux)

mse_MARS_flux <- mean((modelEval_MARS_flux$Actual - modelEval_MARS_flux$Predicted)^2)
rmse_MARS_flux <- sqrt(mse_MARS_flux)
rmse_MARS_flux
0.1415192

hyper_grid_mars2 <- expand.grid(
  degree = 1:5, 
  nprune = seq(2, 30, length.out = 5) %>% floor()
  )

set.seed(42)

# cross validated model
mars2_flux <- train(
  x = x_CO2_Flux_train,
  y = y_CO2_Flux_train,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneGrid = hyper_grid_mars2
)

# best model
mars2_flux$bestTune
   nprune degree
12      9      3

ggplot(mars2_flux)



preds_MARS_flux <- predict(mars2_flux, testSet_flux)

modelEval_MARS_flux <- cbind(testSet_flux$CO2_Flux, preds_MARS_flux)
colnames(modelEval_MARS_flux) <- c('Actual', 'Predicted')
modelEval_MARS_flux <- as.data.frame(modelEval_MARS_flux)

mse_MARS_flux <- mean((modelEval_MARS_flux$Actual - modelEval_MARS_flux$Predicted)^2)
rmse_MARS_flux <- sqrt(mse_MARS_flux)
rmse_MARS_flux
0.1415192


GAM_flux <- gam(CO2_Flux ~ s(Water_Content,k=3)+s(Incubation_Time,k=3)+s(Org_C,k=3)+s(N_Total,k=3)+s(CN,k=3), 
					data = trainSet_flux, family=gaussian, method="REML")
preds_GAM_flux <- predict(GAM_flux, testSet_flux)

modelEval_GAM_flux <- cbind(testSet_flux$CO2_Flux, preds_GAM_flux)
colnames(modelEval_GAM_flux) <- c('Actual', 'Predicted')
modelEval_GAM_flux <- as.data.frame(modelEval_GAM_flux)

mse_GAM_flux <- mean((modelEval_GAM_flux$Actual - modelEval_GAM_flux$Predicted)^2)
rmse_GAM_flux <- sqrt(mse_GAM_flux)
rmse_GAM_flux
0.1853081


set.seed(42)

# cross validated model
gam1_flux <- train(
  x = x_CO2_Flux_train,
  y = y_CO2_Flux_train,
  method = "gam",
  family = "gaussian",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneLength = 50
)

preds_GAM_flux <- predict(gam1_flux, testSet_flux)

modelEval_GAM_flux <- cbind(testSet_flux$CO2_Flux, preds_GAM_flux)
colnames(modelEval_GAM_flux) <- c('Actual', 'Predicted')
modelEval_GAM_flux <- as.data.frame(modelEval_GAM_flux)

mse_GAM_flux <- mean((modelEval_GAM_flux$Actual - modelEval_GAM_flux$Predicted)^2)
rmse_GAM_flux <- sqrt(mse_GAM_flux)
rmse_GAM_flux
0.2150264  -----> lebih jelek



hyper_grid_TR_FLUX <- expand.grid(
  minsplit = seq(5, 25, 1),
  maxdepth = seq(3, 25, 1)
)

models_TR_FLUX <- list()

for (i in 1:nrow(hyper_grid_TR_FLUX)) {
  minsplit <- hyper_grid_TR_FLUX$minsplit[i]
  maxdepth <- hyper_grid_TR_FLUX$maxdepth[i]

  models_TR_FLUX[[i]] <- rpart(
    formula = CO2_Flux ~.,
    data    = trainSet_flux,
    method  = "anova", 
    control = list(minsplit = minsplit, maxdepth = maxdepth)
    )
}


get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}
hyper_grid_TR_FLUX %>%
  mutate(
    cp    = purrr::map_dbl(models_TR_FLUX, get_cp),
    error = purrr::map_dbl(models_TR_FLUX, get_min_error)
    ) %>%
  arrange(error) %>%
  top_n(-10, wt = error)
   minsplit maxdepth         cp     error
1        23        6 0.01000000 0.7125692
2        10        9 0.04269116 0.7484777
3         5        6 0.04269116 0.7705659
4        24       23 0.01000000 0.7743415
5         9       16 0.04269116 0.7800527
6         8       20 0.04269116 0.7838889
7         9       11 0.04269116 0.7952695
8        11       12 0.04269116 0.7953476
9         5       18 0.04269116 0.7993317
10       23       11 0.01000000 0.7993986

TR_FLUX_final <- rpart(
    formula = CO2_Flux ~ .,
    data    = trainSet_flux,
    method  = "anova",
    control = list(minsplit = 23, maxdepth = 6, cp = 0.01)
    )


preds_TR_FLUX <- predict(TR_FLUX_final, testSet_flux)

modelEval_TR_FLUX <- cbind(testSet_flux$CO2_Flux, preds_TR_FLUX)
colnames(modelEval_TR_FLUX) <- c('Actual', 'Predicted')
modelEval_TR_FLUX <- as.data.frame(modelEval_TR_FLUX)

mse_TR_FLUX <- mean((modelEval_TR_FLUX$Actual - modelEval_TR_FLUX$Predicted)^2)
rmse_TR_FLUX <- sqrt(mse_TR_FLUX)
rmse_TR_FLUX
 0.1633684

set.seed(42)
tr_caret <- train(
  x = x_CO2_Flux_train,
  y = y_CO2_Flux_train,
  method = "rpart",
  tuneLength = 100, 
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))


preds_TR_FLUX_caret <- predict(tr_caret, testSet_flux)

modelEval_TR_FLUX_caret <- cbind(testSet_flux$CO2_Flux, preds_TR_FLUX_caret)
colnames(modelEval_TR_FLUX_caret) <- c('Actual', 'Predicted')
modelEval_TR_FLUX_caret <- as.data.frame(modelEval_TR_FLUX_caret)

mse_TR_FLUX_caret <- mean((modelEval_TR_FLUX_caret$Actual - modelEval_TR_FLUX_caret$Predicted)^2)
rmse_TR_FLUX_caret <- sqrt(mse_TR_FLUX_caret)
rmse_TR_FLUX_caret
0.1633684




set.seed(42)
model_rf <- train(
		  x = x_CO2_Flux_train,
		  y = y_CO2_Flux_train,
          method = "rf", 
          metric = "RMSE",
          tuneLength = 200,
          trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))


model_rf
		Random Forest 

		42 samples
		 5 predictor

		No pre-processing
		Resampling: Cross-Validated (10 fold, repeated 10 times) 
		Summary of sample sizes: 38, 38, 39, 37, 36, 38, ... 
		Resampling results across tuning parameters:

		  mtry  RMSE       Rsquared   MAE      
		  2     0.2137909  0.5268412  0.1863954
		  3     0.2099061  0.5444497  0.1835807
		  4     0.2064931  0.5556005  0.1806206
		  5     0.2040709  0.5692018  0.1778408

		RMSE was used to select the optimal model using the smallest value.
		The final value used for the model was mtry = 5.


preds_RF_FLUX <- predict(model_rf, testSet_flux)

modelEval_RF_FLUX <- cbind(testSet_flux$CO2_Flux, preds_RF_FLUX)
colnames(modelEval_RF_FLUX) <- c('Actual', 'Predicted')
modelEval_RF_FLUX <- as.data.frame(modelEval_RF_FLUX)

mse_RF_FLUX <- mean((modelEval_RF_FLUX$Actual - modelEval_RF_FLUX$Predicted)^2)
rmse_RF_FLUX <- sqrt(mse_RF_FLUX)
rmse_RF_FLUX
0.1326669



hyper_grid_RF_FLUX <- expand.grid(
  mtry       = seq(2, 5, by = 1),
  node_size  = seq(3, 20, by = 2),
  num.trees	 = c(750, 1000, 2500, 5000, 7500),
  OOB_RMSE   = 0
)



for(i in 1:nrow(hyper_grid_RF_FLUX)) {
  
  # train model
  model_rf_FLUX <- ranger(
    formula         = CO2_Flux ~., 
    data            = scaled_syva, 
    num.trees       = hyper_grid_RF_FLUX$num.trees[i],
    mtry            = hyper_grid_RF_FLUX$mtry[i],
    min.node.size   = hyper_grid_RF_FLUX$node_size[i],
	sample.fraction = .70,
    seed            = 42
    )
  
  # add OOB error to grid
  hyper_grid_RF_FLUX$OOB_RMSE[i] <- sqrt(model_rf_FLUX$prediction.error)
}

hyper_grid_RF_FLUX %>% 
  dplyr::arrange(OOB_RMSE) %>%
  
 arrange(OOB_RMSE) %>%
  top_n(-10, wt = OOB_RMSE)
		   mtry node_size num.trees  OOB_RMSE
		1     5         3      2500 0.1852111
		2     5         3      5000 0.1857110
		3     5         3      7500 0.1859855
		4     5         3       750 0.1861499
		5     4         3       750 0.1874259
		6     5         3      1000 0.1875399
		7     4         3      1000 0.1882716
		8     4         3      2500 0.1886846
		9     4         3      5000 0.1891030
		10    4         3      7500 0.1894188


RF_FLUX_Final<- 	randomForest(
		CO2_Flux ~ .,
		data		= trainSet_flux,
		ntree   	= 2500,
		mtry		= 5,
		nodesize	= 3,
		importance	= TRUE)
		
preds_RF_FLUX <- predict(RF_FLUX_Final, testSet_flux)

modelEval_RF_FLUX <- cbind(testSet_flux$CO2_Flux, preds_RF_FLUX)
colnames(modelEval_RF_FLUX) <- c('Actual', 'Predicted')
modelEval_RF_FLUX <- as.data.frame(modelEval_RF_FLUX)

mse_RF_FLUX <- mean((modelEval_RF_FLUX$Actual - modelEval_RF_FLUX$Predicted)^2)
rmse_RF_FLUX <- sqrt(mse_RF_FLUX)
rmse_RF_FLUX
0.1337143  --> lebih jelek



svm1 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train,
		method = "svmLinear", 
		metric = "RMSE",
		trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10),
		tuneLength = 200)


svm2 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train,
		method = "svmRadial", 
		metric = "RMSE",
		trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10),
		tuneLength = 200)

		
		
		
res1 <-as_tibble(svm1$results[which.min(svm1$results[,2]),])
res2 <-as_tibble(svm2$results[which.min(svm2$results[,2]),])	


df_svm <-tibble(Model=c('SVM Linear','SVM Radial' ),Accuracy=c(res1$RMSE ,res2$RMSE ))
df_svm %>% arrange(Accuracy)
# A tibble: 2 x 2
  Model      Accuracy
  <chr>         <dbl>
1 SVM Radial    0.248
2 SVM Linear    0.254

preds_SVR_FLUX <- predict(svm1, x_CO2_Flux_train)

modelEval_SVR_FLUX <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX)
colnames(modelEval_SVR_FLUX) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX <- as.data.frame(modelEval_SVR_FLUX)

mse_SVR_FLUX <- mean((modelEval_SVR_FLUX$Actual - modelEval_SVR_FLUX$Predicted)^2)
rmse_SVR_FLUX <- sqrt(mse_SVR_FLUX)
rmse_SVR_FLUX
0.2607305

preds_SVR_FLUX <- predict(svm2, x_CO2_Flux_train)

modelEval_SVR_FLUX <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX)
colnames(modelEval_SVR_FLUX) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX <- as.data.frame(modelEval_SVR_FLUX)

mse_SVR_FLUX <- mean((modelEval_SVR_FLUX$Actual - modelEval_SVR_FLUX$Predicted)^2)
rmse_SVR_FLUX <- sqrt(mse_SVR_FLUX)
rmse_SVR_FLUX
0.2989397


SVR_FLUX_tune <- tune.svm(CO2_Flux ~., 
			data = trainSet_flux, 
              gamma = seq(0.01,2,by=0.1), 
			  cost = 2^(2:4),
			  kernel="sigmoid", #"linear", "polynomial","radial", "sigmoid"
              tunecontrol = tune.control(sampling = "fix")
             )

SVR_FLUX_tune1:linear 

Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.01    4

- best performance: 0.04369968 

> SVR_FLUX_tune2: polynomial

Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.21    8

- best performance: 0.1150756 

> SVR_FLUX_tune3:radial

Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.21   16

- best performance: 0.05086158 

> SVR_FLUX_tune4:sigmoid

Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.01   16

- best performance: 0.1078025 



SVR_FLUX_final1 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux,
			kernel = "linear",  ##best kernel untuk data syva
			gamma	= 0.01,
			cost	= 4)
			

preds_SVR_FLUX <- predict(SVR_FLUX_final1, testSet_flux)

modelEval_SVR_FLUX <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX)
colnames(modelEval_SVR_FLUX) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX <- as.data.frame(modelEval_SVR_FLUX)

mse_SVR_FLUX <- mean((modelEval_SVR_FLUX$Actual - modelEval_SVR_FLUX$Predicted)^2)
rmse_SVR_FLUX <- sqrt(mse_SVR_FLUX)
rmse_SVR_FLUX
0.2514202

SVR_FLUX_final2 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux,
			kernel = "polynomial",  ##best kernel untuk data syva
			gamma	= 0.21,
			cost	= 8)
			

preds_SVR_FLUX2 <- predict(SVR_FLUX_final2, testSet_flux)

modelEval_SVR_FLUX2 <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX2)
colnames(modelEval_SVR_FLUX2) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX2 <- as.data.frame(modelEval_SVR_FLUX2)

mse_SVR_FLUX2 <- mean((modelEval_SVR_FLUX2$Actual - modelEval_SVR_FLUX2$Predicted)^2)
rmse_SVR_FLUX2 <- sqrt(mse_SVR_FLUX2)
rmse_SVR_FLUX2
0.2468237

SVR_FLUX_final3 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux,
			kernel = "radial",  ##best kernel untuk data syva
			gamma	= 0.21,
			cost	= 16)
			

preds_SVR_FLUX3 <- predict(SVR_FLUX_final3, testSet_flux)

modelEval_SVR_FLUX3 <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX3)
colnames(modelEval_SVR_FLUX3) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX3 <- as.data.frame(modelEval_SVR_FLUX3)

mse_SVR_FLUX3 <- mean((modelEval_SVR_FLUX3$Actual - modelEval_SVR_FLUX3$Predicted)^2)
rmse_SVR_FLUX3 <- sqrt(mse_SVR_FLUX3)
rmse_SVR_FLUX3
0.2458408

SVR_FLUX_final4 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux,
			kernel = "sigmoid",  ##best kernel untuk data syva
			gamma	= 0.01,
			cost	= 16)
			

preds_SVR_FLUX4 <- predict(SVR_FLUX_final4, testSet_flux)

modelEval_SVR_FLUX4 <- cbind(testSet_flux$CO2_Flux, preds_SVR_FLUX4)
colnames(modelEval_SVR_FLUX4) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX4 <- as.data.frame(modelEval_SVR_FLUX4)

mse_SVR_FLUX4 <- mean((modelEval_SVR_FLUX4$Actual - modelEval_SVR_FLUX4$Predicted)^2)
rmse_SVR_FLUX4 <- sqrt(mse_SVR_FLUX4)
rmse_SVR_FLUX4
0.2213268   --->> paling oke


hyper_grid_FLUX_GBM <- expand.grid(
	shrinkage = c(.01, .1, .3),
	interaction.depth = c(1, 3, 5, 6),
	n.minobsinnode = c(5, 10, 15),
	bag.fraction = c(.8, .9, 1), 
	optimal_trees = 0,               # a place to dump results
	min_RMSE = 0                     # a place to dump results
)

random_index_FLUX_GBM  <- sample(1:nrow(trainSet_flux), nrow(trainSet_flux))

random_FLUX_GBM_train <- trainSet_flux[random_index_FLUX_GBM, ]

# grid search 
for(i in 1:nrow(hyper_grid_FLUX_GBM)) {
  
  # reproducibility
  set.seed(42)
  
  # train model
  gbm_tune_FLUX <- gbm(
    formula = CO2_Flux ~ .,
    distribution = "gaussian",
    data = scaled_syva,
    n.trees = 1500,
    interaction.depth = hyper_grid_FLUX_GBM$interaction.depth[i],
    shrinkage = hyper_grid_FLUX_GBM$shrinkage[i],
    n.minobsinnode = hyper_grid_FLUX_GBM$n.minobsinnode[i],
    bag.fraction = hyper_grid_FLUX_GBM$bag.fraction[i],
    train.fraction = .70,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid_FLUX_GBM$optimal_trees[i] <- which.min(gbm_tune_FLUX$valid.error)
  hyper_grid_FLUX_GBM$min_RMSE[i] <- sqrt(min(gbm_tune_FLUX$valid.error))
}

hyper_grid_FLUX_GBM %>% 
  dplyr::arrange(min_RMSE) %>%
    top_n(-10, wt = min_RMSE)
   shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees  min_RMSE
1       0.10                 1             15          0.9          1334 0.1670058
2       0.10                 3             15          0.9          1334 0.1670058
3       0.10                 5             15          0.9          1334 0.1670058
4       0.10                 6             15          0.9          1334 0.1670058
5       0.30                 1             15          0.9           377 0.1677870
6       0.30                 3             15          0.9           377 0.1677870
7       0.30                 5             15          0.9           377 0.1677870
8       0.30                 6             15          0.9           377 0.1677870
9       0.01                 1             15          0.9         14794 0.1705962
10      0.01                 3             15          0.9         14794 0.1705962
11      0.01                 5             15          0.9         14794 0.1705962
12      0.01                 6             15          0.9         14794 0.1705962


GBM_FLUX <- gbm(
	formula 			= CO2_Flux ~ .,
	data 				= trainSet_flux,
	distribution 		= "gaussian",
	n.trees 			= 1334,
	interaction.depth 	= 1,
	shrinkage 			= 0.1,
	n.minobsinnode 		= 15,
	bag.fraction 		= 0.9, 
	cv					= 10,
	n.cores 			= NULL, # will use all cores by default
	verbose 			= FALSE
	)

preds_GBM_FLUX <- predict(GBM_FLUX, testSet_flux)

modelEval_GBM_FLUX <- cbind(testSet_flux$CO2_Flux, preds_GBM_FLUX)
colnames(modelEval_GBM_FLUX) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX <- as.data.frame(modelEval_GBM_FLUX)

mse_GBM_FLUX <- mean((modelEval_GBM_FLUX$Actual - modelEval_GBM_FLUX$Predicted)^2)
rmse_GBM_FLUX <- sqrt(mse_GBM_FLUX)
rmse_GBM_FLUX


GBM_FLUX <- gbm(
	formula 			= CO2_Flux ~ .,
	data 				= trainSet_flux,
	distribution 		= "gaussian",
	n.trees 			= 17,
	interaction.depth 	= 6,
	shrinkage 			= 0.3,
	n.minobsinnode 		= 5,
	bag.fraction 		= 1.00, 
	cv					= 10,
	n.cores 			= NULL, # will use all cores by default
	verbose 			= FALSE
	)
	


preds_GBM_FLUX <- predict(GBM_FLUX, testSet_flux)

modelEval_GBM_FLUX <- cbind(testSet_flux$CO2_Flux, preds_GBM_FLUX)
colnames(modelEval_GBM_FLUX) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX <- as.data.frame(modelEval_GBM_FLUX)

mse_GBM_FLUX <- mean((modelEval_GBM_FLUX$Actual - modelEval_GBM_FLUX$Predicted)^2)
rmse_GBM_FLUX <- sqrt(mse_GBM_FLUX)
rmse_GBM_FLUX
0.1529021


gbmGrid <-  expand.grid(interaction.depth = c(5, 9), 
                        n.trees = (1:100), 
                        shrinkage = c(0.1, 0.3),
                        n.minobsinnode = c(5, 6, 9))
                        
nrow(gbmGrid)

set.seed(42)
gbmFit2 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train, 
                 method = "gbm", 
                 trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10), 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)


gbmFit2$bestTune
    n.trees interaction.depth shrinkage n.minobsinnode
639      39                 9       0.1              1

gbmGrid <-  expand.grid(interaction.depth = 9, 
                        n.trees = 39, 
                        shrinkage = 0.1,
                        n.minobsinnode = 1)
						
set.seed(42)
gbmFit2 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train, 
                 method = "gbm", 
                 trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10), 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
				 
preds_GBM_FLUX <- predict(gbmFit2, testSet_flux)

modelEval_GBM_FLUX <- cbind(testSet_flux$CO2_Flux, preds_GBM_FLUX)
colnames(modelEval_GBM_FLUX) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX <- as.data.frame(modelEval_GBM_FLUX)

mse_GBM_FLUX <- mean((modelEval_GBM_FLUX$Actual - modelEval_GBM_FLUX$Predicted)^2)
rmse_GBM_FLUX <- sqrt(mse_GBM_FLUX)
rmse_GBM_FLUX
0.162281
 
 
 
set.seed(42)

grid <-  expand.grid(layer1 = c(1:50),
                     layer2 = 0,
                     layer3 = 0)

NN_FLUX1 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train,
            method = "neuralnet", 
            tuneGrid = grid,
            metric = "RMSE",
            trControl = trainControl(
              method = "repeatedcv",
              number = 10,
			  repeats = 10,
              verboseIter = TRUE)
            )



preds_NN_FLUX1 <- predict(NN_FLUX1, testSet_flux)

modelEval_NN_FLUX1 <- cbind(testSet_flux$CO2_Flux, preds_NN_FLUX1)
colnames(modelEval_NN_FLUX1) <- c('Actual', 'Predicted')
modelEval_NN_FLUX1 <- as.data.frame(modelEval_NN_FLUX1)

mse_NN_FLUX1 <- mean((modelEval_NN_FLUX1$Actual - modelEval_NN_FLUX1$Predicted)^2)
rmse_NN_FLUX1 <- sqrt(mse_NN_FLUX1)
rmse_NN_FLUX1
0.1314307

NN_FLUX1$bestTune
   layer1 layer2 layer3
26     26      0      0

grid2 <-  expand.grid(layer1 = c(1:50),
                     layer2 = c(1:50),
                     layer3 = 0)

NN_FLUX2 <- train(
		x = x_CO2_Flux_train,
		y = y_CO2_Flux_train,
            method = "neuralnet", 
            tuneGrid = grid2,
            metric = "RMSE",
            trControl = trainControl(
              method = "repeatedcv",
              number = 10,
			  repeats = 10,
              verboseIter = TRUE)
            )


preds_NN_FLUX2 <- predict(NN_FLUX2, testSet_flux)

modelEval_NN_FLUX2 <- cbind(testSet_flux$CO2_Flux, preds_NN_FLUX2)
colnames(modelEval_NN_FLUX2) <- c('Actual', 'Predicted')
modelEval_NN_FLUX2 <- as.data.frame(modelEval_NN_FLUX2)

mse_NN_FLUX2 <- mean((modelEval_NN_FLUX2$Actual - modelEval_NN_FLUX2$Predicted)^2)
rmse_NN_FLUX2 <- sqrt(mse_NN_FLUX2)
rmse_NN_FLUX2
0.1119248

NN_FLUX2$bestTune
     layer1 layer2 layer3
1955     40      5      0



library(readxl)
library(readxl)
Data_S_WL <- read_excel("D:/Publikasi/Pak Heru B Pulunggono/7 Simulating and Modeling CO2 flux emited from decomposed oil palm root - HBP, SF, DN, SA, LLN, MZF/Data/Data_S.xlsx", 
    sheet = "Sheet3")
View(Data_S_WL)

View(Data_S_WL)
str(Data_S_WL)

tibble [55 x 7] (S3: tbl_df/tbl/data.frame)
 $ Water_Content  : num [1:55] 15 15 15 15 15 15 15 15 15 15 ...
 $ Incubation_Time: num [1:55] 1 1 1 1 1 2 2 2 2 2 ...
 $ CO2_Flux       : num [1:55] 0.0761 0.0426 0.1101 0.0648 0.0834 ...
 $ Org_C          : num [1:55] 55.4 55.4 54.9 54.9 55.2 ...
 $ N_Total        : num [1:55] 0.457 0.402 0.402 0.429 0.456 ...
 $ CN             : num [1:55] 121 138 137 128 121 ...
 $ Weight_loss    : num [1:55] 0.035 0.3648 0.079 0.0999 0.02 ...

Data_S_WL <- na.omit(Data_S_WL) #buang kolom kosong

maxs1 <- apply(Data_S_WL, 2, max) 
mins1 <- apply(Data_S_WL, 2, min)
scaled_syva1 <- as.data.frame(scale(Data_S_WL, center = mins1, 
                              scale = maxs1 - mins1))	

set.seed(42)

sampleSplit_flux1 <- sample.split(Y=scaled_syva1$CO2_Flux, SplitRatio=0.7)
trainSet_flux1 <- subset(x=scaled_syva1, sampleSplit_flux1==TRUE)
testSet_flux1 <- subset(x=scaled_syva1, sampleSplit_flux1==FALSE)

x_CO2_Flux_train1 = subset(trainSet_flux1, select = -CO2_Flux) %>% as.matrix() 
y_CO2_Flux_train1 = trainSet_flux1$CO2_Flux

x_CO2_Flux_test1 = subset(testSet_flux1, select = -CO2_Flux) %>% as.matrix() 
y_CO2_Flux_test1 = testSet_flux1$CO2_Flux



set.seed = 42
LM_flux_1 <- caret::train(
  x = x_CO2_Flux_train1,
  y = y_CO2_Flux_train1,
  method = "lm",
  family = "gaussian",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10)
)

summary(LM_flux_1)
		Call:
		lm(formula = .outcome ~ ., data = dat, family = "gaussian")

		Residuals:
			 Min       1Q   Median       3Q      Max 
		-0.37747 -0.15147 -0.00564  0.09085  0.60083 

		Coefficients:
						Estimate Std. Error t value Pr(>|t|)    
		(Intercept)      -1.0301     1.3580  -0.759   0.4538    
		Water_Content    -0.1013     0.1166  -0.869   0.3914    
		Incubation_Time  -0.7586     0.1605  -4.727  4.7e-05 ***
		Org_C            -0.3693     0.2738  -1.349   0.1873    
		N_Total           2.1568     1.5420   1.399   0.1718    
		CN                2.0599     1.5040   1.370   0.1807    
		Weight_loss       0.6582     0.3131   2.102   0.0438 *  
		---
		Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

		Residual standard error: 0.2227 on 31 degrees of freedom
		Multiple R-squared:  0.4845,	Adjusted R-squared:  0.3848 
		F-statistic: 4.857 on 6 and 31 DF,  p-value: 0.001333

preds_LM_flux1 <- predict(LM_flux_1, testSet_flux1)

modelEval_LM_flux1 <- cbind(testSet_flux1$CO2_Flux, preds_LM_flux1)
colnames(modelEval_LM_flux1) <- c('Actual', 'Predicted')
modelEval_LM_flux1 <- as.data.frame(modelEval_LM_flux1)

mse_LM_flux1 <- mean((modelEval_LM_flux1$Actual - modelEval_LM_flux1$Predicted)^2)
rmse_LM_flux1 <- sqrt(mse_LM_flux1)
rmse_LM_flux1
0.2504635



hyper_grid_mars1 <- expand.grid(
  degree = 1:5, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
  )

set.seed(42)

# cross validated model
mars_flux1_1 <- train(
  x = x_CO2_Flux_train1,
  y = y_CO2_Flux_train1,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneGrid = hyper_grid_mars1
)

# best model
mars_flux1_1$bestTune
   nprune degree
22     12      3

ggplot(mars_flux1_1)


preds_MARS_flux1 <- predict(mars_flux1_1, testSet_flux1)

modelEval_MARS_flux1 <- cbind(testSet_flux1$CO2_Flux, preds_MARS_flux1)
colnames(modelEval_MARS_flux1) <- c('Actual', 'Predicted')
modelEval_MARS_flux1 <- as.data.frame(modelEval_MARS_flux1)

mse_MARS_flux1 <- mean((modelEval_MARS_flux1$Actual - modelEval_MARS_flux1$Predicted)^2)
rmse_MARS_flux1 <- sqrt(mse_MARS_flux1)
rmse_MARS_flux1
0.1734524

hyper_grid_mars2_1 <- expand.grid(
  degree = 3:5, 
  nprune = seq(2, 30, length.out = 5) %>% floor()
  )

set.seed(42)

# cross validated model
mars2_flux_1 <- train(
  x = x_CO2_Flux_train1,
  y = y_CO2_Flux_train1,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneGrid = hyper_grid_mars2_1
)

# best model
mars2_flux_1$bestTune


ggplot(mars2_flux_1)
 nprune degree
2      9      3


preds_MARS_flux2_1 <- predict(mars2_flux_1, testSet_flux1)

modelEval_MARS_flux2_1 <- cbind(testSet_flux1$CO2_Flux, preds_MARS_flux2_1)
colnames(modelEval_MARS_flux2_1) <- c('Actual', 'Predicted')
modelEval_MARS_flux2_1 <- as.data.frame(modelEval_MARS_flux2_1)

mse_MARS_flux2_1 <- mean((modelEval_MARS_flux2_1$Actual - modelEval_MARS_flux2_1$Predicted)^2)
rmse_MARS_flux2_1 <- sqrt(mse_MARS_flux2_1)
rmse_MARS_flux2_1
0.1734524



GAM_flux1 <- gam(CO2_Flux ~ s(Water_Content,k=3)+s(Incubation_Time,k=3)+s(Org_C,k=3)+s(N_Total,k=3)+s(CN,k=3)+s(Weight_loss,k=3), 
					data = trainSet_flux1, family=gaussian, method="REML")
preds_GAM_flux1 <- predict(GAM_flux1, testSet_flux1)

modelEval_GAM_flux1 <- cbind(testSet_flux1$CO2_Flux, preds_GAM_flux1)
colnames(modelEval_GAM_flux1) <- c('Actual', 'Predicted')
modelEval_GAM_flux1 <- as.data.frame(modelEval_GAM_flux1)

mse_GAM_flux1 <- mean((modelEval_GAM_flux1$Actual - modelEval_GAM_flux1$Predicted)^2)
rmse_GAM_flux1 <- sqrt(mse_GAM_flux1)
rmse_GAM_flux1
0.3151875


set.seed(42)

# cross validated model
gam1_flux1 <- train(
  x = x_CO2_Flux_train1,
  y = y_CO2_Flux_train1,
  method = "gam",
  family = "gaussian",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats=10),
  tuneLength = 50
)

preds_GAM_flux2_1 <- predict(gam1_flux1, testSet_flux1)

modelEval_GAM_flux2_1 <- cbind(testSet_flux1$CO2_Flux, preds_GAM_flux2_1)
colnames(modelEval_GAM_flux2_1) <- c('Actual', 'Predicted')
modelEval_GAM_flux2_1 <- as.data.frame(modelEval_GAM_flux2_1)

mse_GAM_flux2_1 <- mean((modelEval_GAM_flux2_1$Actual - modelEval_GAM_flux2_1$Predicted)^2)
rmse_GAM_flux2_1 <- sqrt(mse_GAM_flux2_1)
rmse_GAM_flux2_1




hyper_grid_TR_FLUX1 <- expand.grid(
  minsplit = seq(5, 25, 1),
  maxdepth = seq(3, 25, 1)
)

models_TR_FLUX1 <- list()

for (i in 1:nrow(hyper_grid_TR_FLUX1)) {
  minsplit1 <- hyper_grid_TR_FLUX1$minsplit[i]
  maxdepth1 <- hyper_grid_TR_FLUX1$maxdepth[i]

  models_TR_FLUX1[[i]] <- rpart(
    formula = CO2_Flux ~.,
    data    = trainSet_flux1,
    method  = "anova", 
    control = list(minsplit = minsplit1, maxdepth = maxdepth1)
    )
}


get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}
hyper_grid_TR_FLUX1 %>%
  mutate(
    cp    = purrr::map_dbl(models_TR_FLUX1, get_cp),
    error = purrr::map_dbl(models_TR_FLUX1, get_min_error)
    ) %>%
  arrange(error) %>%
  top_n(-10, wt = error)
		   minsplit maxdepth   cp     error
		1        12       23 0.01 0.3996688
		2        12        8 0.01 0.4045873
		3        14       21 0.01 0.4064480
		4        15       22 0.01 0.4066825
		5        13        3 0.01 0.4085323
		6         9       12 0.01 0.4101695
		7         8       14 0.01 0.4103862
		8         9       10 0.01 0.4107661
		9        15       24 0.01 0.4119931
		10        8       21 0.01 0.4121272

TR_FLUX_final1 <- rpart(
    formula = CO2_Flux ~ .,
    data    = trainSet_flux1,
    method  = "anova",
    control = list(minsplit = 12, maxdepth = 23, cp = 0.03996688)
    )


preds_TR_FLUX1 <- predict(TR_FLUX_final1, testSet_flux1)

modelEval_TR_FLUX1 <- cbind(testSet_flux1$CO2_Flux, preds_TR_FLUX1)
colnames(modelEval_TR_FLUX1) <- c('Actual', 'Predicted')
modelEval_TR_FLUX1 <- as.data.frame(modelEval_TR_FLUX1)

mse_TR_FLUX1 <- mean((modelEval_TR_FLUX1$Actual - modelEval_TR_FLUX1$Predicted)^2)
rmse_TR_FLUX1 <- sqrt(mse_TR_FLUX1)
rmse_TR_FLUX1
0.1691769


set.seed(42)
tr_caret1 <- train(
  x = x_CO2_Flux_train1,
  y = y_CO2_Flux_train1,
  method = "rpart",
  tuneLength = 250, 
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))


preds_TR_FLUX_caret1 <- predict(tr_caret1, testSet_flux1)

modelEval_TR_FLUX_caret2_1 <- cbind(testSet_flux1$CO2_Flux, preds_TR_FLUX_caret1)
colnames(modelEval_TR_FLUX_caret2_1) <- c('Actual', 'Predicted')
modelEval_TR_FLUX_caret2_1 <- as.data.frame(modelEval_TR_FLUX_caret2_1)

mse_TR_FLUX_caret2_1 <- mean((modelEval_TR_FLUX_caret2_1$Actual - modelEval_TR_FLUX_caret2_1$Predicted)^2)
rmse_TR_FLUX_caret2_1 <- sqrt(mse_TR_FLUX_caret2_1)
rmse_TR_FLUX_caret2_1
0.1798438




set.seed(42)
model_rf1 <- train(
		  x = x_CO2_Flux_train1,
		  y = y_CO2_Flux_train1,
          method = "rf", 
          metric = "RMSE",
          tuneLength = 250,
          trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))

preds_RF_FLUX1 <- predict(model_rf1, testSet_flux1)

modelEval_RF_FLUX1 <- cbind(testSet_flux1$CO2_Flux, preds_RF_FLUX1)
colnames(modelEval_RF_FLUX1) <- c('Actual', 'Predicted')
modelEval_RF_FLUX1 <- as.data.frame(modelEval_RF_FLUX1)

mse_RF_FLUX1 <- mean((modelEval_RF_FLUX1$Actual - modelEval_RF_FLUX1$Predicted)^2)
rmse_RF_FLUX1 <- sqrt(mse_RF_FLUX1)
rmse_RF_FLUX1
0.1594484



hyper_grid_RF_FLUX1 <- expand.grid(
  mtry       = seq(2, 5, by = 1),
  node_size  = seq(3, 20, by = 2),
  num.trees	 = c(750, 1000, 2500, 5000, 7500),
  OOB_RMSE   = 0
)



for(i in 1:nrow(hyper_grid_RF_FLUX1)) {
  
  # train model
  model_rf_FLUX1 <- ranger(
    formula         = CO2_Flux ~., 
    data            = scaled_syva1, 
    num.trees       = hyper_grid_RF_FLUX1$num.trees[i],
    mtry            = hyper_grid_RF_FLUX1$mtry[i],
    min.node.size   = hyper_grid_RF_FLUX1$node_size[i],
	sample.fraction = .70,
    seed            = 42
    )
  
  # add OOB error to grid
  hyper_grid_RF_FLUX1$OOB_RMSE[i] <- sqrt(model_rf_FLUX1$prediction.error)
}

hyper_grid_RF_FLUX1 %>% 
  dplyr::arrange(OOB_RMSE) %>%
  
 arrange(OOB_RMSE) %>%
  top_n(-10, wt = OOB_RMSE)
		   mtry node_size num.trees  OOB_RMSE
		1     5         3      2500 0.1417813
		2     5         3      5000 0.1419753
		3     5         3      7500 0.1421889
		4     5         3      1000 0.1445193
		5     5         5      2500 0.1445745
		6     5         3       750 0.1447075
		7     5         5      5000 0.1449959
		8     5         5      7500 0.1451489
		9     5         5      1000 0.1476309
		10    5         5       750 0.1480521


RF_FLUX_Final1 <- 	randomForest(
		CO2_Flux ~ .,
		data		= trainSet_flux1,
		ntree   	= 2500,
		mtry		= 5,
		nodesize	= 3,
		importance	= TRUE)
		
preds_RF_FLUX2_1 <- predict(RF_FLUX_Final1, testSet_flux1)

modelEval_RF_FLUX2_1 <- cbind(testSet_flux1$CO2_Flux, preds_RF_FLUX2_1)
colnames(modelEval_RF_FLUX2_1) <- c('Actual', 'Predicted')
modelEval_RF_FLUX2_1 <- as.data.frame(modelEval_RF_FLUX2_1)

mse_RF_FLUX2_1 <- mean((modelEval_RF_FLUX2_1$Actual - modelEval_RF_FLUX2_1$Predicted)^2)
rmse_RF_FLUX2_1 <- sqrt(mse_RF_FLUX2_1)
rmse_RF_FLUX2_1
0.1587987 ## paling oke



svm1_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1,
		method = "svmLinear", 
		metric = "RMSE",
		trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10),
		tuneLength = 200)


svm2_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1,
		method = "svmRadial", 
		metric = "RMSE",
		trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10),
		tuneLength = 200)

		
res1_1 <-as_tibble(svm1_1$results[which.min(svm1_1$results[,2]),])
res2_1 <-as_tibble(svm2_1$results[which.min(svm2_1$results[,2]),])	


df_svm1 <-tibble(Model=c('SVM Linear','SVM Radial' ),Accuracy=c(res1_1$RMSE ,res2_1$RMSE ))
df_svm1 %>% arrange(Accuracy)
# A tibble: 2 x 2
  Model      Accuracy
  <chr>         <dbl>
1 SVM Linear    0.216
2 SVM Radial    0.250


preds_SVR_FLUX1 <- predict(svm1_1, x_CO2_Flux_train1)

modelEval_SVR_FLUX1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX1)
colnames(modelEval_SVR_FLUX1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX1 <- as.data.frame(modelEval_SVR_FLUX1)

mse_SVR_FLUX1 <- mean((modelEval_SVR_FLUX1$Actual - modelEval_SVR_FLUX1$Predicted)^2)
rmse_SVR_FLUX1 <- sqrt(mse_SVR_FLUX1)
rmse_SVR_FLUX1
0.2579971

preds_SVR_FLUX2_1 <- predict(svm2_1, x_CO2_Flux_train1)

modelEval_SVR_FLUX2_1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX2_1)
colnames(modelEval_SVR_FLUX2_1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX2_1 <- as.data.frame(modelEval_SVR_FLUX2_1)

mse_SVR_FLUX2_1 <- mean((modelEval_SVR_FLUX2_1$Actual - modelEval_SVR_FLUX2_1$Predicted)^2)
rmse_SVR_FLUX2_1 <- sqrt(mse_SVR_FLUX2_1)
rmse_SVR_FLUX2_1
0.2313952



SVR_FLUX_tune1_1 <- tune.svm(CO2_Flux ~., 
			data = trainSet_flux1, 
              gamma = seq(0.01,2,by=0.1), 
			  cost = 2^(2:4),
			  kernel="linear", #"linear", "polynomial","radial", "sigmoid"
              tunecontrol = tune.control(sampling = "fix")
             )

SVR_FLUX_tune1_1
Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.01    4

- best performance: 0.06596181 


SVR_FLUX_tune2_1 <- tune.svm(CO2_Flux ~., 
			data = trainSet_flux1, 
              gamma = seq(0.01,2,by=0.1), 
			  cost = 2^(2:4),
			  kernel="polynomial", #"linear", "polynomial","radial", "sigmoid"
              tunecontrol = tune.control(sampling = "fix"))
SVR_FLUX_tune2_1
Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.11    4

- best performance: 0.04207823 


SVR_FLUX_tune3_1 <- tune.svm(CO2_Flux ~., 
			data = trainSet_flux1, 
              gamma = seq(0.01,2,by=0.1), 
			  cost = 2^(2:4),
			  kernel="radial", #"linear", "polynomial","radial", "sigmoid"
              tunecontrol = tune.control(sampling = "fix"))
SVR_FLUX_tune3_1
Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.91    4

- best performance: 0.04104527 



SVR_FLUX_tune4_1 <- tune.svm(CO2_Flux ~., 
			data = trainSet_flux1, 
              gamma = seq(0.01,2,by=0.1), 
			  cost = 2^(2:4),
			  kernel="sigmoid", #"linear", "polynomial","radial", "sigmoid"
              tunecontrol = tune.control(sampling = "fix"))
SVR_FLUX_tune4_1
Parameter tuning of ‘svm’:

- sampling method: fixed training/validation set 

- best parameters:
 gamma cost
  0.01   16

- best performance: 0.04278077 


SVR_FLUX_final1_1 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux1,
			kernel = "linear",  ##best kernel untuk data syva
			gamma	= 0.01,
			cost	= 4)
			

preds_SVR_FLUX_f1_1 <- predict(SVR_FLUX_final1_1, testSet_flux1)

modelEval_SVR_FLUX_f1_1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX_f1_1)
colnames(modelEval_SVR_FLUX_f1_1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX_f1_1 <- as.data.frame(modelEval_SVR_FLUX_f1_1)

mse_SVR_FLUX_f1_1 <- mean((modelEval_SVR_FLUX_f1_1$Actual - modelEval_SVR_FLUX_f1_1$Predicted)^2)
rmse_SVR_FLUX_f1_1 <- sqrt(mse_SVR_FLUX_f1_1)
rmse_SVR_FLUX_f1_1
0.2262826

SVR_FLUX_final2_1 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux1,
			kernel = "polynomial",  ##best kernel untuk data syva
			gamma	= 0.11,
			cost	= 4)
			

preds_SVR_FLUX2_f1_1 <- predict(SVR_FLUX_final2_1, testSet_flux1)

modelEval_SVR_FLUX2_f1_1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX2_f1_1)
colnames(modelEval_SVR_FLUX2_f1_1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX2_f1_1 <- as.data.frame(modelEval_SVR_FLUX2_f1_1)

mse_SVR_FLUX2_f1_1 <- mean((modelEval_SVR_FLUX2_f1_1$Actual - modelEval_SVR_FLUX2_f1_1$Predicted)^2)
rmse_SVR_FLUX2_f1_1 <- sqrt(mse_SVR_FLUX2_f1_1)
rmse_SVR_FLUX2_f1_1
0.4347594

SVR_FLUX_final3_1 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux1,
			kernel = "radial",  ##best kernel untuk data syva
			gamma	= 0.91,
			cost	= 4)
			

preds_SVR_FLUX3_f1_1 <- predict(SVR_FLUX_final3_1, testSet_flux1)

modelEval_SVR_FLUX3_f1_1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX3_f1_1)
colnames(modelEval_SVR_FLUX3_f1_1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX3_f1_1 <- as.data.frame(modelEval_SVR_FLUX3_f1_1)

mse_SVR_FLUX3_f1_1 <- mean((modelEval_SVR_FLUX3_f1_1$Actual - modelEval_SVR_FLUX3_f1_1$Predicted)^2)
rmse_SVR_FLUX3_f1_1 <- sqrt(mse_SVR_FLUX3_f1_1)
rmse_SVR_FLUX3_f1_1
0.1547514   ##Paling oke

SVR_FLUX_final4_1 <- svm(
			CO2_Flux ~., 
			data	= trainSet_flux1,
			kernel = "sigmoid",  ##best kernel untuk data syva
			gamma	= 0.01,
			cost	= 16)
			

preds_SVR_FLUX4_f1_1 <- predict(SVR_FLUX_final4_1, testSet_flux1)

modelEval_SVR_FLUX4_f1_1 <- cbind(testSet_flux1$CO2_Flux, preds_SVR_FLUX4_f1_1)
colnames(modelEval_SVR_FLUX4_f1_1) <- c('Actual', 'Predicted')
modelEval_SVR_FLUX4_f1_1 <- as.data.frame(modelEval_SVR_FLUX4_f1_1)

mse_SVR_FLUX4_f1_1 <- mean((modelEval_SVR_FLUX4_f1_1$Actual - modelEval_SVR_FLUX4_f1_1$Predicted)^2)
rmse_SVR_FLUX4_f1_1 <- sqrt(mse_SVR_FLUX4_f1_1)
rmse_SVR_FLUX4_f1_1
0.1651917


hyper_grid_FLUX_GBM1 <- expand.grid(
	shrinkage = c(.01, .1, .3),
	interaction.depth = c(1, 3, 5, 6),
	n.minobsinnode = c(5, 10),
	bag.fraction = c(.9, 1), 
	optimal_trees = 0,               # a place to dump results
	min_RMSE = 0                     # a place to dump results
)

random_index_FLUX_GBM1  <- sample(1:nrow(trainSet_flux1), nrow(trainSet_flux1))

random_FLUX_GBM_train1 <- trainSet_flux1[random_index_FLUX_GBM1, ]

# grid search 
for(i in 1:nrow(hyper_grid_FLUX_GBM1)) {
  
  # reproducibility
  set.seed(42)
  
  # train model
  gbm_tune_FLUX1 <- gbm(
    formula = CO2_Flux ~ .,
    distribution = "gaussian",
    data = scaled_syva1,
    n.trees = 1500,
    interaction.depth = hyper_grid_FLUX_GBM1$interaction.depth[i],
    shrinkage = hyper_grid_FLUX_GBM1$shrinkage[i],
    n.minobsinnode = hyper_grid_FLUX_GBM1$n.minobsinnode[i],
    bag.fraction = hyper_grid_FLUX_GBM1$bag.fraction[i],
    train.fraction = .70,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid_FLUX_GBM1$optimal_trees[i] <- which.min(gbm_tune_FLUX1$valid.error)
  hyper_grid_FLUX_GBM1$min_RMSE[i] <- sqrt(min(gbm_tune_FLUX1$valid.error))
}

hyper_grid_FLUX_GBM1 %>% 
  dplyr::arrange(min_RMSE) %>%
    top_n(-10, wt = min_RMSE)
   shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees  min_RMSE
1        0.3                 3             10          0.9           798 0.1784524
2        0.3                 5             10          0.9           798 0.1784524
3        0.3                 6             10          0.9           798 0.1784524
4        0.1                 3             10          0.9          1500 0.1803590
5        0.1                 5             10          0.9          1500 0.1803590
6        0.1                 6             10          0.9          1500 0.1803590
7        0.1                 1             10          0.9           936 0.1929130
8        0.3                 3             10          1.0          1484 0.1942211
9        0.3                 5             10          1.0          1484 0.1942211
10       0.3                 6             10          1.0          1484 0.1942211


set.seed(42)
GBM_FLUX1 <- gbm(
	formula 			= CO2_Flux ~ .,
	data 				= trainSet_flux1,
	distribution 		= "gaussian",
	n.trees 			= 798,
	interaction.depth 	= 3,
	shrinkage 			= 0.3,
	n.minobsinnode 		= 10,
	bag.fraction 		= 0.9, 
	cv					= 10,
	n.cores 			= NULL, # will use all cores by default
	verbose 			= FALSE
	)

preds_GBM_FLUX1 <- predict(GBM_FLUX1, testSet_flux1)

modelEval_GBM_FLUX1 <- cbind(testSet_flux1$CO2_Flux, preds_GBM_FLUX1)
colnames(modelEval_GBM_FLUX1) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX1 <- as.data.frame(modelEval_GBM_FLUX1)

mse_GBM_FLUX1 <- mean((modelEval_GBM_FLUX1$Actual - modelEval_GBM_FLUX1$Predicted)^2)
rmse_GBM_FLUX1 <- sqrt(mse_GBM_FLUX1)
rmse_GBM_FLUX1
0.1689035



gbmGrid1 <-  expand.grid(interaction.depth = c(3,5, 9), 
                        n.trees = (1:100), 
                        shrinkage = c(0.1, 0.3),
                        n.minobsinnode = c(5, 6, 9, 10))
                        
nrow(gbmGrid1)

set.seed(42)
gbmFit2_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1, 
                 method = "gbm", 
                 trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10), 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid1)


gbmFit2_1$bestTune
     n.trees interaction.depth shrinkage n.minobsinnode
1749      49                 5       0.3              6

preds_GBM_FLUX1_1 <- predict(gbmFit2_1, testSet_flux1)

modelEval_GBM_FLUX1_1 <- cbind(testSet_flux1$CO2_Flux, preds_GBM_FLUX1_1)
colnames(modelEval_GBM_FLUX1_1) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX1_1 <- as.data.frame(modelEval_GBM_FLUX1_1)

mse_GBM_FLUX1_1 <- mean((modelEval_GBM_FLUX1_1$Actual - modelEval_GBM_FLUX1_1$Predicted)^2)
rmse_GBM_FLUX1_1 <- sqrt(mse_GBM_FLUX1_1)
rmse_GBM_FLUX1_1
0.149238

gbmGrid1_1 <-  expand.grid(interaction.depth = c(3,5, 6), 
                        n.trees = (1:100), 
                        shrinkage = c(0.01, 0.1),
                        n.minobsinnode = c(5, 6, 7, 9))
                        
nrow(gbmGrid1_1)


						
set.seed(42)
gbmFit3_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1, 
                 method = "gbm", 
                 trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10), 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid1_1)

gbmFit3_1$bestTune
     n.trees interaction.depth shrinkage n.minobsinnode
2300     100                 6       0.1              7

preds_GBM_FLUX2_1 <- predict(gbmFit3_1, testSet_flux1)

modelEval_GBM_FLUX2_1 <- cbind(testSet_flux1$CO2_Flux, preds_GBM_FLUX2_1)
colnames(modelEval_GBM_FLUX2_1) <- c('Actual', 'Predicted')
modelEval_GBM_FLUX2_1 <- as.data.frame(modelEval_GBM_FLUX2_1)

mse_GBM_FLUX2_1 <- mean((modelEval_GBM_FLUX2_1$Actual - modelEval_GBM_FLUX2_1$Predicted)^2)
rmse_GBM_FLUX2_1 <- sqrt(mse_GBM_FLUX2_1)
rmse_GBM_FLUX2_1
0.1655159
 
 
set.seed(42)

grid_NN1 <-  expand.grid(layer1 = c(1:50),
                     layer2 = 0,
                     layer3 = 0)

set.seed(42)
NN_FLUX1_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1,
            method = "neuralnet", 
            tuneGrid = grid_NN1,
            metric = "RMSE",
            trControl = trainControl(
              method = "repeatedcv",
              number = 10,
			  repeats = 10,
              verboseIter = TRUE)
            )



preds_NN_FLUX1_1 <- predict(NN_FLUX1_1, testSet_flux1)

modelEval_NN_FLUX1_1 <- cbind(testSet_flux1$CO2_Flux, preds_NN_FLUX1_1)
colnames(modelEval_NN_FLUX1_1) <- c('Actual', 'Predicted')
modelEval_NN_FLUX1_1 <- as.data.frame(modelEval_NN_FLUX1_1)

mse_NN_FLUX1_1 <- mean((modelEval_NN_FLUX1_1$Actual - modelEval_NN_FLUX1_1$Predicted)^2)
rmse_NN_FLUX1_1 <- sqrt(mse_NN_FLUX1_1)
rmse_NN_FLUX1_1
0.2480498

NN_FLUX1_1$bestTune
  layer1 layer2 layer3
1      1      0      0

grid_NN2 <-  expand.grid(layer1 = c(1:10),
                     layer2 = c(1:10),
                     layer3 = 0)

NN_FLUX2_1 <- train(
		x = x_CO2_Flux_train1,
		y = y_CO2_Flux_train1,
            method = "neuralnet", 
            tuneGrid = grid_NN2,
            metric = "RMSE",
            trControl = trainControl(
              method = "repeatedcv",
              number = 10,
			  repeats = 10,
              verboseIter = TRUE)
            )


preds_NN_FLUX2_1 <- predict(NN_FLUX2_1, testSet_flux1)

modelEval_NN_FLUX2_1 <- cbind(testSet_flux1$CO2_Flux, preds_NN_FLUX2_1)
colnames(modelEval_NN_FLUX2_1) <- c('Actual', 'Predicted')
modelEval_NN_FLUX2_1 <- as.data.frame(modelEval_NN_FLUX2_1)

mse_NN_FLUX2_1 <- mean((modelEval_NN_FLUX2_1$Actual - modelEval_NN_FLUX2_1$Predicted)^2)
rmse_NN_FLUX2_1 <- sqrt(mse_NN_FLUX2_1)
rmse_NN_FLUX2_1
0.3626356

NN_FLUX2_1$bestTune
   layer1 layer2 layer3
51      6      1      0


## ---- Plotting actual with modeled ----- ##

theme1 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=6),
				axis.text.x=element_text(size=6),
				axis.title.y=element_text(size=7),
				axis.title.x=element_text(size=7))

theme2 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=6),
				axis.text.x=element_text(size=6),
				axis.title.y=element_blank(),
				axis.title.x=element_text(size=7))

theme3 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=6),
				axis.text.x=element_text(size=6),
				axis.title.y=element_text(size=7),
				axis.title.x=element_blank())

theme4 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=6),
				axis.text.x=element_text(size=6),
				axis.title.y=element_blank(),
				axis.title.x=element_blank())


MLR_abline <- lm(Actual ~ Predicted, modelEval_LM_flux) ##fitting line 
MLR_plot <- ggplot(modelEval_LM_flux, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(MLR_abline)[["Predicted"]], 
              intercept = coef(MLR_abline)[["(Intercept)"]])+
  ggtitle("MLR") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)


MARS_abline <- lm(Actual ~ Predicted, modelEval_MARS_flux) ##fitting line 
MARS_plot <- ggplot(modelEval_MARS_flux, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(MARS_abline)[["Predicted"]], 
              intercept = coef(MARS_abline)[["(Intercept)"]])+
  ggtitle("MARS") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)
					

GAM_abline <- lm(Actual ~ Predicted, modelEval_GAM_flux) ##fitting line 
GAM_plot <- ggplot(modelEval_GAM_flux, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) +  
  geom_abline(color='red', size=0.6, slope = coef(GAM_abline)[["Predicted"]], 
              intercept = coef(GAM_abline)[["(Intercept)"]])+
  ggtitle("GAM") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)					
					

TR_abline <- lm(Actual ~ Predicted, modelEval_TR_FLUX) ##fitting line 
TR_plot <- ggplot(modelEval_TR_FLUX, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) +  
  geom_abline(color='red', size=0.6, slope = coef(TR_abline)[["Predicted"]], 
              intercept = coef(TR_abline)[["(Intercept)"]])+
  ggtitle("TR") + theme_bw() + theme3 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)


RF_abline <- lm(Actual ~ Predicted, modelEval_RF_FLUX) ##fitting line 
RF_plot <- ggplot(modelEval_RF_FLUX, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(RF_abline)[["Predicted"]], 
              intercept = coef(RF_abline)[["(Intercept)"]])+
  ggtitle("RF") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)


SVR_abline <- lm(Actual ~ Predicted, modelEval_SVR_FLUX) ##fitting line 
SVR_plot <- ggplot(modelEval_SVR_FLUX, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(SVR_abline)[["Predicted"]], 
              intercept = coef(SVR_abline)[["(Intercept)"]])+
  ggtitle("SVR") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)



GBM_abline <- lm(Actual ~ Predicted, modelEval_GBM_FLUX) ##fitting line 
GBM_plot <- ggplot(modelEval_GBM_FLUX, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(GBM_abline)[["Predicted"]], 
              intercept = coef(GBM_abline)[["(Intercept)"]])+
  ggtitle("GBM") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)


NN1_abline <- lm(Actual ~ Predicted, modelEval_NN_FLUX1) ##fitting line 
NN1_plot <- ggplot(modelEval_NN_FLUX1, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(NN1_abline)[["Predicted"]], 
              intercept = coef(NN1_abline)[["(Intercept)"]])+
  ggtitle("NN1") + theme_bw() + theme2 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)


NN2_abline <- lm(Actual ~ Predicted, modelEval_NN_FLUX2) ##fitting line 
NN2_plot <- ggplot(modelEval_NN_FLUX2, aes(x=Predicted, y= Actual)) +
  geom_point() + 
  #geom_smooth(method=lm , color="black", fill="#69b3a2", se=TRUE) +
  #ylim(c(0, 0.25)) +
  labs(x=expression(Predicted~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")"), y=expression(Observed~CO[2]~Flux~"("~mg~m^{-2}~~sec^{-1}~")")) + 
  geom_abline(color='red', size=0.6, slope = coef(NN2_abline)[["Predicted"]], 
              intercept = coef(NN2_abline)[["(Intercept)"]])+
  ggtitle("NN2") + theme_bw() + theme4 +
  stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "*`,`~")),
					label.y= Inf, label.x = Inf, vjust = 1, hjust = 1.1, size = 4)




MLR_plot+MARS_plot+GAM_plot+TR_plot+RF_plot+SVR_plot+GBM_plot+NN1_plot+NN2_plot+plot_layout(ncol=3)


## ------- calculating variable importance ------##

y <- scaled_syva$CO2_Flux
X <- scaled_syva[-which(names(scaled_syva) == "CO2_Flux")]

mod_mlr <- Predictor$new(LM_flux, data = X, y = y)
mod_mars <- Predictor$new(mars_flux, data = X, y = y)
mod_gam <- Predictor$new(GAM_flux, data = X, y = y)
mod_tr <- Predictor$new(TR_FLUX_final, data = X, y = y)
mod_rf <- Predictor$new(RF_FLUX_Final, data = X, y = y)
mod_svr <- Predictor$new(SVR_FLUX_final, data = X, y = y)
mod_gbm <- Predictor$new(GBM_FLUX, data = X, y = y)
mod_nn1 <- Predictor$new(NN_FLUX1, data = X, y = y)
mod_nn2 <- Predictor$new(NN_FLUX2, data = X, y = y)



imp.mlr <- FeatureImp$new(mod_mlr, loss = "mse")
imp.mars <- FeatureImp$new(mod_mars, loss = "mse")
imp.gam <- FeatureImp$new(mod_gam, loss = "mse")
imp.tr <- FeatureImp$new(mod_tr, loss = "mse")
imp.rf <- FeatureImp$new(mod_rf, loss = "mse")
imp.svr <- FeatureImp$new(mod_svr, loss = "mse")
imp.gbm <- FeatureImp$new(mod_gbm, loss = "mse")
imp.nn1 <- FeatureImp$new(mod_nn1, loss = "mse")
imp.nn2 <- FeatureImp$new(mod_nn2, loss = "mse")

theme1 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=7),
				axis.text.x=element_text(size=7),
				axis.title.y=element_text(size=8),
				axis.title.x=element_text(size=8))

theme2 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=7),
				axis.text.x=element_text(size=7),
				axis.title.y=element_blank(),
				axis.title.x=element_text(size=8))

theme3 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=7),
				axis.text.x=element_text(size=7),
				axis.title.y=element_text(size=8),
				axis.title.x=element_blank())

theme4 <- theme(title =element_text(size=8, face='bold'),
				axis.text.y=element_text(size=7),
				axis.text.x=element_text(size=7),
				axis.title.y=element_blank(),
				axis.title.x=element_blank())
				
# plot output
p1 <- plot(imp.mlr) + ggtitle("MLR")+theme_bw()+theme3
p2 <- plot(imp.mars) + ggtitle("MARS")+theme_bw()+theme4
p3 <- plot(imp.gam) + ggtitle("GAM")+theme_bw()+theme4
p4 <- plot(imp.tr) + ggtitle("TR")+theme_bw()+theme3
p5 <- plot(imp.rf) + ggtitle("RF")+theme_bw()+theme4
p6 <- plot(imp.svr) + ggtitle("SVR")+theme_bw()+theme4
p7 <- plot(imp.gbm) + ggtitle("GBM")+theme_bw()+theme4
p8 <- plot(imp.nn1) + ggtitle("NN1")+theme_bw()+theme2
p9 <- plot(imp.nn2) + ggtitle("NN2")+theme_bw()+theme4

gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, nrow = 3)

p_fix <- p1+p2+p3+p4+p5+p6+p7+p8+p9
p_fix+plot_layout(nrow=3)

imp_lm <- calc.relimp(LM_flux, type = c("lmg"), rela = TRUE) 

			Response variable: CO2_Flux 
			Total response variance: 0.01511992 
			Analysis based on 56 observations 

			5 Regressors: 
			Water_Content Incubation_Time Org_C N_Total CN 
			Proportion of variance explained by model: 14.89%
			Metrics are normalized to sum to 100% (rela=TRUE). 

			Relative importance metrics: 

								   lmg
			Water_Content   0.27114114
			Incubation_Time 0.18339021
			Org_C           0.08860245
			N_Total         0.19553839
			CN              0.26132781

			Average coefficients for different model sizes: 

									 1X         2Xs
			Water_Content   -0.04248802 -0.05852178
			Incubation_Time -0.07599868 -0.06968247
			Org_C            0.07905991  0.05272527
			N_Total         -0.14775892 -0.01023641
			CN               0.18955553  0.30803254
									3Xs         4Xs
			Water_Content   -0.06854621 -0.07708813
			Incubation_Time -0.06052368 -0.07212689
			Org_C            0.01236797 -0.04447864
			N_Total          0.09283263  0.16912462
			CN               0.39033752  0.43914898
									5Xs
			Water_Content   -0.08040713
			Incubation_Time -0.08925513
			Org_C           -0.10939525
			N_Total          0.36790625
			CN               0.61059937


mars_imp <- vi(mars_flux)
		# A tibble: 5 x 2
		  Variable        Importance
		  <chr>                <dbl>
		1 CN                       5
		2 Incubation_Time          4
		3 Org_C                    4
		4 Water_Content            0
		5 N_Total                  0




y <- scaled_syva$CO2_Flux
X <- scaled_syva[-which(names(scaled_syva) == "CO2_Flux")]
mod_gam <- Predictor$new(GAM_flux, data = X, y = y)

imp_gam <- FeatureImp$new(mod_gam, loss = "mae")
			Interpretation method:  FeatureImp 
			error function: mae

			Analysed predictor: 
			Prediction task: unknown 


			Analysed data:
			Sampling from data.frame with 80 rows and 5 columns.

			Head of results:
					  feature importance.05 importance
			1              CN      2.580722   2.888552
			2           Org_C      2.451717   2.760379
			3 Incubation_Time      2.469231   2.677141
			4         N_Total      2.134309   2.341465
			5   Water_Content      1.062238   1.072480
			  importance.95 permutation.error
			1      2.944363        0.15992813
			2      2.843731        0.15283167
			3      2.809959        0.14822311
			4      2.452274        0.12963797
			5      1.115918        0.05937911


plot(imp_gam)+ggtitle()


TR_finalPlot <- as.data.frame(TR_FLUX_final$variable.importance)
TR_finalPlot
        TR_FLUX_final$variable.importance
CN                             0.10525357
N_Total                        0.09021735



RF_imp <-vi(RF_FLUX_Final)
		# A tibble: 5 x 2
		  Variable        Importance
		  <chr>                <dbl>
		1 Org_C                13.0 
		2 Water_Content        12.6 
		3 Incubation_Time       4.81
		4 CN                   -3.68
		5 N_Total              -7.96



w_flux_SVR <- t(SVR_FLUX_final$coefs) %*% SVR_FLUX_final$SV                 # weight vectors
w_flux_SVR <- apply(w_flux_SVR, 2, function(v){sqrt(sum(v^2))})  			# weight
w_flux_SVR <- sort(w_flux_SVR, decreasing = T)
print(w_flux_SVR)

Incubation_Time           Org_C         N_Total 
     0.09423262      0.06286242      0.04912947 
  Water_Content              CN 
     0.04566156      0.01831627 


summary(GBM_FLUX)
                            var    rel.inf
Org_C                     Org_C 38.5627539
CN                           CN 22.9444639
N_Total                 N_Total 20.5419244
Water_Content     Water_Content 17.2101421
Incubation_Time Incubation_Time  0.7407157



olden(NN_FLUX1, bar_plot=FALSE)
                importance
Water_Content   -1.7835155
Incubation_Time -1.5265944
Org_C           -1.9908934
N_Total         -0.9998446
CN              -1.0539569

olden(NN_FLUX2, bar_plot=FALSE)
                importance
Water_Content    -3.132195
Incubation_Time  -4.900666
Org_C            -3.349891
N_Total          -2.491434
CN                5.076444
