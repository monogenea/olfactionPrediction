
# Mon Oct 29 13:17:55 2018 ------------------------------
## DREAM olfaction prediction challenge
library(caret)
#library(yardstick)
library(rsample)
library(tidyverse)
library(recipes)
library(magrittr)

# Create directory and download files
dir.create("data/")
ghurl <- "https://github.com/dream-olfaction/olfaction-prediction/raw/master/data/"
download.file(paste0(ghurl, "TrainSet.txt"),
              destfile = "data/trainSet.txt")
# download.file(paste0(ghurl, "TestSet.txt"),
#               destfile = "data/testSet.txt")
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
X <-  mutate(molFeats, Y = meanPlsnt$pleasantness)

# Clean up
idx <- nearZeroVar(X, uniqueCut = .5)
X <- X[,-idx]

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

# Design recipe
myRec <- recipe(Y ~ ., data = trainSet) %>% 
      step_knnimpute(all_predictors(), K = 5) %>% 
      step_nzv(all_predictors()) %>%
      step_YeoJohnson(all_numeric()) %>% 
      step_center(all_predictors()) %>% 
      step_scale(all_predictors())
      
# Train PLS model
plsMod <- train(myRec, data = trainSet,
                method = "pls",
                tuneGrid = data.frame(ncomp = seq(5, 20, length.out = 8)))

# Vaidate on testset
preds <- predict(plsMod, newdata = testSet)
plot(preds, testSet$Y)