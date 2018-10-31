
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
      filter(Dilution == "1/1,000") # take one dilution
molFeats <- read_tsv("data/DRAGON.txt") # molecular features

# Determine intersection of compounds in features and responses
commonMols <- intersect(responses$CID,
                        molFeats$CID)
# Subset features and responses accordingly
responses %<>% filter(CID %in% commonMols)
molFeats %<>% filter(CID %in% commonMols)

# Compute median pleasantness across the population
medianPlsnt <- responses %>% 
      group_by(CID) %>% 
      summarise(pleasantness = median(`VALENCE/PLEASANTNESS`, na.rm = T))
all(medianPlsnt$CID == molFeats$CID) # TRUE - rownames match

# Concatenate predictors (molFeats) and population pleasantness
X <-  mutate(molFeats, Y = medianPlsnt$pleasantness) %>% 
      select(-CID)

# Filter nzv
X <- X[,-nearZeroVar(X, freqCut = 4)] # == 80/20

# Split train/test with rsample
set.seed(100)
initSplit <- initial_split(X, prop = .9)
trainSet <- training(initSplit)
testSet <- testing(initSplit)

# Create 5-fold cross-validation, convert to caret class
set.seed(100)
myFolds <- vfold_cv(trainSet, v = 5) %>% 
      rsample2caret()
ctrl <- trainControl(method = "cv")
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

plsMod <- train(myRec, data = trainSet,
                method = "pls",
                tuneGrid = data.frame(ncomp = c(5, 10, 15, 20, 30)),
                trControl = ctrl)

rfMod <- train(myRec, data = trainSet,
               method = "ranger",
               tuneGrid = expand.grid(mtry = seq(10, 2000, length.out = 5),
                                      splitrule = c("variance", "extratrees"),
                                      min.node.size = c(5, 10)),
               num.trees = 1000,
               trControl = ctrl)

svmMod <- train(myRec, data = trainSet,
                method = "svmPoly",
                tuneLength = 3,
                trControl = ctrl)

# Validate on testset
preds <- predict(plsMod, newdata = testSet)
plot(preds, testSet$Y,
     xlim = c(25,80), ylim = c(25,80),
     xlab = "Predicted", ylab = "Observed")
abline(a=0, b=1)

