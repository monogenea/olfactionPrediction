
# Mon Oct 29 13:17:55 2018 ------------------------------
## DREAM olfaction prediction challenge
library(caret)
library(rsample)
library(tidyverse)
library(recipes)
library(magrittr)
library(doMC)

# Create directory and download files
dir.create("data/")
ghurl <- "https://github.com/dream-olfaction/olfaction-prediction/raw/master/data/"
download.file(paste0(ghurl, "TrainSet.txt"),
              destfile = "data/trainSet.txt")
download.file(paste0(ghurl, "molecular_descriptors_data.txt"),
              destfile = "data/molFeats.txt")

# Read files with readr, select least diluted compounds            
responses <- read_tsv("data/trainSet.txt") %>%
      rename(CID = `Compound Identifier`)

molFeats <- read_tsv("data/molFeats.txt") # molecular features

# Determine intersection of compounds in features and responses
commonMols <- intersect(responses$CID,
                        molFeats$CID)
# Subset features and responses accordingly
responses %<>% filter(CID %in% commonMols)
molFeats %<>% filter(CID %in% commonMols)

# Compute median pleasantness across the population
medianPlsnt <- responses %>% 
      group_by(CID) %>% 
      dplyr::summarise(pleasantness = median(`VALENCE/PLEASANTNESS`, na.rm = T))
all(medianPlsnt$CID == molFeats$CID) # TRUE - rownames match

# Concatenate predictors (molFeats) and population pleasantness
X <-  mutate(molFeats, Y = medianPlsnt$pleasantness) %>% 
      select(-CID)

# Filter nzv
X <- X[,-nearZeroVar(X, freqCut = 4)] # == 80/20

# Split train/test with rsample
set.seed(100)
initSplit <- initial_split(X, prop = .9,
                           strata = "Y")
trainSet <- training(initSplit)
testSet <- testing(initSplit)

# Create 5-fold cross-validation, convert to caret class
set.seed(100)
myFolds <- vfold_cv(trainSet, v = 5, repeats = 5,
                    strata = "Y") %>% 
      rsample2caret()
ctrl <- trainControl(method = "cv",
                     selectionFunction = "oneSE")
ctrl$index <- myFolds$index
ctrl$indexOut <- myFolds$indexOut

# binary vars
binVars <- which(sapply(X, function(x){all(x %in% 0:1)}))
missingVars <- which(apply(X, 2, function(k){any(is.na(k))}))

# Design recipe
myRec <- recipe(Y ~ ., data = trainSet) %>% 
      step_YeoJohnson(all_predictors(), -binVars) %>% 
      step_center(all_predictors(), -binVars) %>% 
      step_scale(all_predictors(), -binVars) %>% 
      step_meanimpute(missingVars)

# simple PCA, plot
pcaRec <- myRec %>% 
      step_pca(all_predictors())

myPCA <- prep(pcaRec, training = trainSet, retain = T) %>% 
      juice()
colGrad <- trainSet$Y/100 # add color

plot(myPCA$PC1, myPCA$PC2,
     col = rgb(1 - colGrad, 0, colGrad,.5),
     pch = 16, xlab = "PC1", ylab = "PC2")
legend("topleft", pch = 16,
       col = rgb(c(0,.5,1), 0, c(1,.5,0), alpha = .5), 
       legend = c("Pleasant", "Neutral", "Unpleasant"), bty = "n")

# Train
doMC::registerDoMC(10)
knnMod <- train(myRec, data = trainSet,
           method = "knn",
           tuneGrid = data.frame(k = seq(5, 25, by = 4)),
           trControl = ctrl)

enetMod <- train(myRec, data = trainSet,
                 method = "glmnet",
                 tuneGrid = expand.grid(alpha = seq(0, 1, length.out = 5),
                                        lambda = seq(.5, 2, length.out = 5)),
                 trControl = ctrl)

svmMod <- train(myRec, data = trainSet,
                method = "svmRadial",
                tuneLength = 8,
                trControl = ctrl)

rfMod <- train(myRec, data = trainSet,
               method = "ranger",
               tuneLength = 8,
               num.trees = 5000,
               trControl = ctrl)

xgbMod <- train(myRec, data = trainSet,
                method = "xgbTree",
                tuneLength = 8,
                trControl = ctrl)

cubMod <- train(myRec, data = trainSet,
                method = "cubist",
                tuneLength = 8,
                trControl = ctrl)

modelList <- list("KNN" = knnMod,
                  "ENET" = enetMod,
                  "SVM" = svmMod,
                  "RF" = rfMod,
                  "XGB" = xgbMod,
                  "CUB" = cubMod)

bwplot(resamples(modelList),
       metric = "RMSE")

# Validate on test set with ensemble
allPreds <- sapply(modelList, predict, newdata = testSet)
ensemblePred <- rowSums(allPreds) / length(modelList)

# Plot predicted vs. observed; create PNG
plot(ensemblePred, testSet$Y,
     xlim = c(0,100), ylim = c(0,100),
     xlab = "Predicted", ylab = "Observed",
     pch = 16, col = rgb(0, 0, 0, .25))
abline(a=0, b=1)

writeLines(capture.output(sessionInfo()), "sessionInfo")
