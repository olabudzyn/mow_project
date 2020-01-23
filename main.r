# install.packages("utiml")
# install.packages("RWeka")
# 
# library("utiml")
# library("rJava")
# library("RWeka")
# library(mldr)

library("mlr")
library("BBmisc")
library("mldr")
library("ParamHelpers")

# myyeast <- read.arff("yeast")
# myscene <- read.arff("scene")
# 
# yeast <- read.arff("yeast")
# 
yeast_train <- read.arff("yeast-train",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")
yeast_test <- read.arff("yeast-test",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")
# 
# my_yeast_train <- yeast_train[["dataframe"]]
# yeast_train_label <- my_yeast_train[yeast_train[["labelIndices"]]]
# yeast_train_attr <- my_yeast_train[1:103]
# 
# my_yeast_test <- yeast_test[["dataframe"]]
# yeast_test_label <- my_yeast_test[yeast_test[["labelIndices"]]]
# yeast_test_attr <- my_yeast_test[1:103]

set.seed(1729)

# wczytanie danych prawidlowe
#yeast <- read.arff("yeast")

data.train = yeast_train[["dataframe"]]
data.test = yeast_test[["dataframe"]]

target = labels(yeast_train[["attributes"]])
labels = target[yeast_train[["labelIndices"]]]

feats = setdiff(target, labels)
new_data.train = data.train[feats]
new_data.test = data.test[feats]
logic_data <- lapply(data.train[labels], as.logical)
logic_data.test <- lapply(data.test[labels], as.logical)

data_with_logic = cbind(new_data.train, logic_data)
data_with_logic.test = cbind(new_data.test, logic_data.test)

# stworzenie zadania
yeast.task = makeMultilabelTask(id = "multi", data = data_with_logic.test, target = labels)

# podzial danych na testowe i trenujace
# n = getTaskSize(yeast.task)
# train.set = seq(1, n, by = 2)
# test.set = seq(2, n, by = 2)

#stworzenie klasyfikatorów binarnych 

# drzewo decyzyjne
binary.tree = makeLearner("classif.rpart")

# Naiwny Bayes
binary.naiveBayes = makeLearner("classif.naiveBayes")

# Maszyna wektorów nosnych
binary.svm = makeLearner("classif.svm")

multilabelChain <- function(learner, data, binary.label.data, labels) {
  
  nr_labels = length(labels)
  weightMatrix <- matrix(0, nr_labels, dim(data)[2])
  
  
  for (i in 1 : nr_labels)
  {
    chain = cbind(data, binary.label.data[i])
    binarytask = makeClassifTask(id = "BinaryClassification", data = chain, target = labels[i])
    mod = train(learner, binarytask)
    task.pred = predict(mod, task = binarytask)
    response = getPredictionResponse(task.pred)
    col_name <- sprintf("Class%d", i)
    new_data = cbind(data, lapply(response, as.logical))
    colnames(new_data)[dim(new_data)[2]] <- col_name
    data <- new_data
  }
  
  return(mod)
}


model = multilabelChain(binary.tree, new_data.train, logic_data, labels)
task.pred = predict(model, task = yeast.task)

perceptron <- function(x, y, eta, niter) {
  
  errors <- rep(0, niter)
  weight <- t(runif(dim(x)[2], min = 0, max = 1))
  
  
  for (jj in 1:niter) {
    for (ii in 1:length(y)) {
      
      #sigmoidalna funkcja aktywacji
      beta = 5
      z <-  as.numeric(x[ii, ]) %*% t(weight)
      
      ypred = 1 / (1 + exp (-beta * z))
      
      
      #poprawka wag
      weightdiff <- eta * (y[ii] - ypred) *   as.numeric(x[ii,])
      
      weight <- weight + t(weightdiff)
      
      
      if (abs(ypred > 0.5))
        ypredbin = 1
      else
        ypredbin = 0
      
      #obliczanie ilosci bledow
      if ((y[ii] - ypredbin) != 0.0) {
        errors[jj] <- errors[jj] + 1
      }
      
    }
  }
  
  newList <- list(weight,errors)
  print(newList[1])
  return(newList)
  
 # return(weight)
}




results <- function(weights, x) {
  val <- rep(0, dim(x)[1])
  
  for (ii in 1:dim(x)[1]) {
    beta = 5
    z <-  as.numeric(x[ii, ]) %*% t(weights)
    
    ypred = 1 / (1 + exp (-beta * z))
    
    
    
    if (abs(ypred > 0.5))
      val[ii] = 1
    else
      val[ii] = 0
    
  }
  return(val)
}


# 
# chain <- generateChain(yeast_train_attr, yeast_train_label, 0.001, 5)
# 
# result <- results(chain[14, ], yeast_test_attr)



# plot data - a picture is worth a 1000 words. Melt data => then ggplot
# library(ggplot2)
# ggplot(irissubdf, aes(x = sepal, y = petal)) +
#   geom_point(aes(colour = species, shape = species), size = 3) +
#   xlab("sepal length") +
#   ylab("petal length") +
#   ggtitle("Species vs sepal and petal lengths")
# add binary labels corresponding to species - Initialize all values to 1
# add setosa label of -1. The binary +1, -1 labels are in the fourth
# column. It is better to create two separate data frames: one containing
# the attributes while the other contains the class values.













