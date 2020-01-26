install.packages("utiml")

library("ParamHelpers")
library("mlr")
library("utiml")
#library("BBmisc")
library("mldr")
library(datarobot)
source("multilabelChain.R")

yeast_train <- read.arff("yeast-train",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")
yeast_test <- read.arff("yeast-test",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")

scene_train <- read.arff("scene-train",use_xml=TRUE,auto_extension = TRUE, "scene.xml")
scene_test <- read.arff("scene-test",use_xml=TRUE,auto_extension = TRUE, "scene.xml")

set.seed(1729)

yeast.data.train = yeast_train[["dataframe"]]
yeast.data.test = yeast_test[["dataframe"]]

scene.data.train = scene_train[["dataframe"]]
scene.data.test = scene_test[["dataframe"]]

########## YEAST
yeast.target = labels(yeast_train[["attributes"]])
yeast.labels = yeast.target[yeast_train[["labelIndices"]]]

yeast.feats = setdiff(yeast.target, yeast.labels)
yeast.new_data.train = yeast.data.train[yeast.feats]
yeast.new_data.test = yeast.data.test[yeast.feats]
yeast.logic_data <- lapply(yeast.data.train[yeast.labels], as.logical)
yeast.logic_data.test <- lapply(yeast.data.test[yeast.labels], as.logical)

yeast.data_with_logic = cbind(yeast.new_data.train, yeast.logic_data)
yeast.data_with_logic.test = cbind(yeast.new_data.test, yeast.logic_data.test)


########## SCENE
scene.target = labels(scene_train[["attributes"]])
scene.labels = scene.target[scene_train[["labelIndices"]]]

scene.feats = setdiff(scene.target, scene.labels)
scene.new_data.train = scene.data.train[scene.feats]
scene.new_data.test = scene.data.test[scene.feats]
scene.logic_data <- lapply(scene.data.train[scene.labels], as.logical)
scene.logic_data.test <- lapply(scene.data.test[scene.labels], as.logical)

scene.data_with_logic = cbind(scene.new_data.train, scene.logic_data)
scene.data_with_logic.test = cbind(scene.new_data.test, scene.logic_data.test)


#stworzenie klasyfikatorów binarnych 

# drzewo decyzyjne
binary.tree = makeLearner("classif.rpart")

# Naiwny Bayes
binary.naiveBayes = makeLearner("classif.naiveBayes")

# Maszyna wektorów nosnych
binary.svm = makeLearner("classif.svm")


########################### PRZEWIDYWANIA DLA YEAST ##############################
yeast.labels_data  <- do.call(cbind, yeast.data.test[yeast.labels])


##### results dla klasyfikatora naiveBayes
yeast.model.naiveBayes = multilabelChain(binary.naiveBayes, yeast.new_data.train, yeast.logic_data)
yeast.prediction.naiveBayes = predict.myChainClassier(yeast.model.naiveBayes, yeast.new_data.test, yeast.logic_data.test)

yeast.hammloss.naiveBayes  <- measureMultilabelHamloss(yeast.labels_data, yeast.prediction.naiveBayes)
yeast.subset01.naiveBayes  <- measureMultilabelSubset01(yeast.labels_data, yeast.prediction.naiveBayes)
yeast.f1.naiveBayes <- measureMultilabelF1(yeast.labels_data, yeast.prediction.naiveBayes)
yeast.acc.naiveBayes <- measureMultilabelACC(yeast.labels_data, yeast.prediction.naiveBayes)

##### results dla klasyfikatora tree
yeast.model.tree = multilabelChain(binary.tree, yeast.new_data.train, yeast.logic_data)
yeast.prediction.tree = predict.myChainClassier(yeast.model.tree, yeast.new_data.test, yeast.logic_data.test)

yeast.hammloss.tree <- measureMultilabelHamloss(yeast.labels_data, yeast.prediction.tree)
yeast.subset01.tree  <- measureMultilabelSubset01(yeast.labels_data, yeast.prediction.tree)
yeast.f1.tree <- measureMultilabelF1(yeast.labels_data, yeast.prediction.tree)
yeast.acc.tree  <- measureMultilabelACC(yeast.labels_data, yeast.prediction.tree)


##### results dla klasyfikatora svm
yeast.model.svm = multilabelChain(binary.svm, yeast.new_data.train, yeast.logic_data)
yeast.prediction.svm = predict.myChainClassier(yeast.model.svm, yeast.new_data.test, yeast.logic_data.test)

yeast.hammloss.svm  <- measureMultilabelHamloss(yeast.labels_data, yeast.prediction.svm)
yeast.subset01.svm  <- measureMultilabelSubset01(yeast.labels_data, yeast.prediction.svm)
yeast.f1.svm <- measureMultilabelF1(yeast.labels_data, yeast.prediction.svm)
yeast.acc.svm  <- measureMultilabelACC(yeast.labels_data, yeast.prediction.svm)


########################### PRZEWIDYWANIA DLA SCENE ##############################
scene.labels_data  <- do.call(cbind, scene.data.test[scene.labels])

##### results dla klasyfikatora naiveBayes
scene.model.naiveBayes = multilabelChain(binary.naiveBayes, scene.new_data.train, scene.logic_data)
scene.prediction.naiveBayes = predict.myChainClassier(scene.model.naiveBayes, scene.new_data.test, scene.logic_data.test)

scene.hammloss.naiveBayes <- measureMultilabelHamloss(scene.labels_data, scene.prediction.naiveBayes)
scene.subset01.naiveBayes <- measureMultilabelSubset01(scene.labels_data, scene.prediction.naiveBayes)
scene.f1.naiveBayes <- measureMultilabelF1(scene.labels_data, scene.prediction.naiveBayes)
scene.acc.naiveBayes <- measureMultilabelACC(scene.labels_data, scene.prediction.naiveBayes)


##### results dla klasyfikatora tree
scene.model.tree = multilabelChain(binary.tree, scene.new_data.train, scene.logic_data)
scene.prediction.tree = predict.myChainClassier(scene.model.tree, scene.new_data.test, scene.logic_data.test)

scene.hammloss.tree  <- measureMultilabelHamloss(scene.labels_data, scene.prediction.tree)
scene.subset01.tree  <- measureMultilabelSubset01(scene.labels_data, scene.prediction.tree)
scene.f1.tree <- measureMultilabelF1(scene.labels_data, scene.prediction.tree)
scene.acc.tree  <- measureMultilabelACC(scene.labels_data, scene.prediction.tree)


##### results dla klasyfikatora svm
scene.model.svm = multilabelChain(binary.svm, scene.new_data.train, scene.logic_data)
scene.prediction.svm = predict.myChainClassier(scene.model.svm, scene.new_data.test, scene.logic_data.test)

scene.hammloss.svm  <- measureMultilabelHamloss(scene.labels_data, scene.prediction.svm)
scene.subset01.svm  <- measureMultilabelSubset01(scene.labels_data, scene.prediction.svm)
scene.f1.svm <- measureMultilabelF1(scene.labels_data, scene.prediction.svm)
scene.acc.svm  <- measureMultilabelACC(scene.labels_data, scene.prediction.svm)




