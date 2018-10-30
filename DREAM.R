
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
              destfile = "data/DRAGON.txt")

# Read files with readr, select least diluted compounds            
responses <- read_tsv("data/trainSet.txt") %>%
      rename(CID = `Compound Identifier`) %>%
      filter(Dilution == "1/10") # take the least diluted
molFeats <- read_tsv("data/DRAGON.txt") # molecular features

# Determine intersection of compounds in features and responses
commonMols <- intersect(responses$CID,
                             molFeats$CID)
# Subset features and responses accordingly
responses %<>% filter(CID %in% commonMols)
molFeats %<>% filter(CID %in% commonMols)

# Compute median/mean pleasantness across the population
meanPlsnt <- responses %>% 
      group_by(CID) %>% 
      summarise(pleasantness = median(`VALENCE/PLEASANTNESS`, na.rm = T))
all(meanPlsnt$CID == molFeats$CID) # TRUE - rownames match

# Concatenate predictors (molFeats) and population pleasantness
X <-  mutate(molFeats, Y = meanPlsnt$pleasantness) %>% 
      select(-CID)

# Filter nzv
X <- X[,-nearZeroVar(X, freqCut = 4)] # == 80/20

### DEBUG
X <- select(X, 1:1000, Y)
#####

# Split train/test with rsample
set.seed(100)
initSplit <- initial_split(X, prop = .9)
trainSet <- training(initSplit)
testSet <- testing(initSplit)

# Create 10x 3-fold cross-validation, convert to caret class
set.seed(100)
myFolds <- vfold_cv(trainSet, v = 3, repeats = 10) %>% 
      rsample2caret()
ctrl <- trainControl(method = "cv")
ctrl$index <- myFolds$index
ctrl$indexOut <- myFolds$indexOut

# binary vars
binVars <- which(sapply(X, function(x){all(x %in% 0:1)}))

# Design recipe
myRec <- recipe(Y ~ ., data = trainSet) %>% 
      step_YeoJohnson(all_predictors(), -binVars) %>% 
      step_center(all_predictors(), -binVars) %>% 
      step_scale(all_predictors(), -binVars) %>% 
      step_knnimpute(all_numeric(), K = 5)

test <- prep(myRec, training = trainSet, retain = T)
test <- juice(test)

pcaRec <- myRec %>% 
      step_kpca(all_predictors(), options = list(kernel = "rbfdot",
                                                 kpar = list(sigma = .005)))

myPCA <- prep(pcaRec, training = trainSet, retain = T)
myPCA <- juice(myPCA)
plot(myPCA$kPC1, myPCA$kPC2, cex = trainSet$Y/max(trainSet$Y) + .5,
     col = rgb(0,0,0,.25), pch = 16)

# Train PLS model
doMC::registerDoMC(10)
fdaMod <- train(myRec, data = trainSet,
            method = "rf",
            tuneLength = 4,
            trControl = ctrl)

# Vaidate on testset
preds <- predict(plsMod, newdata = testSet)
plot(preds, testSet$Y)