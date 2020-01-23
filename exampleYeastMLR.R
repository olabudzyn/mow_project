library("mlr")
library("BBmisc")
library("mldr")
library("ParamHelpers")

set.seed(1729)

# wczytanie danych prawidlowe
yeast <- read.arff("yeast")

data = yeast[["dataframe"]]

target = labels(yeast[["attributes"]])
labels = target[yeast[["labelIndices"]]]

feats = setdiff(target, labels)
new_data = data[feats]
logic_data <- lapply(data[labels], as.logical)

# stworzenie zadania
yeast.task = makeMultilabelTask(id = "multi", data = cbind(new_data,logic_data), target = labels)

# podzial danych na testowe i trenujace
n = getTaskSize(yeast.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)


#stworzenie klasyfikatora
lrn.rfsrc = makeLearner("multilabel.randomForestSRC")
lrn.rfsrc

# uczenie => powstaje model
modRandomForestSRC = train(lrn.rfsrc, yeast.task, subset = train.set)

# predykcja dla zestawu testowego
predictRandomForestSRC = predict(modRandomForestSRC, yeast.task, subset=test.set)

#stworzenie klasyfikatorów binarnych 

# drzewo decyzyjne
binary.tree = makeLearner("classif.rpart")

# Naiwny Bayes
binary.naiveBayes = makeLearner("classif.naiveBayes")

# Maszyna wektorów nosnych
binary.svm = makeLearner("classif.svm")

# Lalncuch klasyfikatorow dla binarnych
lrn.cc.tree = makeMultilabelClassifierChainsWrapper(binary.tree)
lrn.cc.naiveBayes = makeMultilabelClassifierChainsWrapper(binary.naiveBayes)
lrn.cc.svm = makeMultilabelClassifierChainsWrapper(binary.svm)

# uczenie => powstaja modele dla chain wrapper
mod.cc.tree = train(lrn.cc.tree, yeast.task, subset = train.set)
mod.cc.naiveBayes = train(lrn.cc.naiveBayes, yeast.task, subset = train.set)
mod.cc.svm= train(lrn.cc.svm, yeast.task, subset = train.set)

# predykcja dla zestawu testowego
predict.cc.tree = predict(mod.cc.tree, task = yeast.task, subset = test.set)
predict.cc.naiveBayes = predict(mod.cc.naiveBayes, task = yeast.task, subset = test.set)
predict.cc.svm = predict(mod.cc.svm, task = yeast.task, subset = test.set)

###### Nested Stacking Wrapper
lrn.ns.tree = makeMultilabelNestedStackingWrapper(binary.tree, order = NULL, cv.folds = 2)
lrn.ns.naiveBayes = makeMultilabelNestedStackingWrapper(binary.naiveBayes, order = NULL, cv.folds = 2)
lrn.ns.svm = makeMultilabelNestedStackingWrapper(binary.svm, order = NULL, cv.folds = 2)

# uczenie => powstaje model dla nested stacking
mod.ns.tree = train(lrn.ns.tree, yeast.task, subset = train.set)
mod.ns.naiveBayes = train(lrn.ns.naiveBayes, yeast.task, subset = train.set)
mod.ns.svm = train(lrn.ns.svm, yeast.task, subset = train.set)

# predykcja dla zestawu testowego
predict.ns.tree = predict(mod.ns.tree, task = yeast.task, subset = test.set)
predict.ns.naiveBayes = predict(mod.ns.naiveBayes, task = yeast.task, subset = test.set)
predict.ns.svm = predict(mod.ns.svm, task = yeast.task, subset = test.set)

listMeasures("multilabel")
measuresList = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc)

# pomiary parametrów
performance(predictRandomForestSRC, measures = measuresList)
performance(predict.cc.tree, measures = measuresList)
performance(predict.ns.tree, measures = measuresList)
performance(predict.cc.naiveBayes, measures = measuresList)
performance(predict.ns.naiveBayes, measures = measuresList)
performance(predict.cc.svm, measures = measuresList)
performance(predict.ns.svm, measures = measuresList)

# resampling
rdesc = makeResampleDesc("CV", iters = 10, split = 2/3)
resampleRansdomForestSRC = resample(lrn.rfsrc, yeast.task, rdesc, measures = measuresList)
resample.cc.tree = resample(lrn.cc.tree, yeast.task, rdesc, measures = measuresList)
resample.ns.tree = resample(lrn.ns.tree, yeast.task, rdesc, measures = measuresList)
resample.cc.naiveBayes = resample(lrn.cc.naiveBayes, yeast.task, rdesc, measures = measuresList)
resample.ns.naiveBayes = resample(lrn.ns.naiveBayes, yeast.task, rdesc, measures = measuresList)
resample.cc.svm = resample(lrn.cc.svm, yeast.task, rdesc, measures = measuresList)
resample.ns.svm = resample(lrn.ns.svm, scene.task, rdesc, measures = measuresList)

mesaure.values <- resampleRansdomForestSRC[["measures.test"]]
mesaure.values.cc.tree <- resample.cc.tree[["measures.test"]]
mesaure.values.ns.tree <- resample.ns.tree[["measures.test"]]
mesaure.values.cc.naiveBayes <- resample.cc.naiveBayes[["measures.test"]]
mesaure.values.ns.naiveBayes <- resample.ns.naiveBayes[["measures.test"]]
mesaure.values.cc.svm <- resample.cc.svm[["measures.test"]]
mesaure.values.ns.svm <- resample.ns.svm[["measures.test"]]

hamloss_mean = mean(mesaure.values[,2])
subset_mean = mean(mesaure.values[,3])
f1_mean = mean(mesaure.values[,4])
acc_mean = mean(mesaure.values[,5])

hamloss_cc_mean.tree = mean(mesaure.values.cc.tree[,2])
subset_cc_mean.tree = mean(mesaure.values.cc.tree[,3])
f1_cc_mean.tree = mean(mesaure.values.cc.tree[,4])
acc_cc_mean.tree = mean(mesaure.values.cc.tree[,5])

hamloss_ns_mean.tree = mean(mesaure.values.ns.tree[,2])
subset_ns_mean.tree = mean(mesaure.values.ns.tree[,3])
f1_ns_mean.tree = mean(mesaure.values.ns.tree[,4])
acc_ns_mean.tree = mean(mesaure.values.ns.tree[,5])

hamloss_cc_mean.naiveBayes = mean(mesaure.values.cc.naiveBayes[,2])
subset_cc_mean.naiveBayes = mean(mesaure.values.cc.naiveBayes[,3])
f1_cc_mean.naiveBayes = mean(mesaure.values.cc.naiveBayes[,4])
acc_cc_mean.naiveBayes = mean(mesaure.values.cc.naiveBayes[,5])

hamloss_ns_mean.naiveBayes = mean(mesaure.values.ns.naiveBayes[,2])
subset_ns_mean.naiveBayes = mean(mesaure.values.ns.naiveBayes[,3])
f1_ns_mean.naiveBayes = mean(mesaure.values.ns.naiveBayes[,4])
acc_ns_mean.naiveBayes = mean(mesaure.values.ns.naiveBayes[,5])

hamloss_cc_mean.svm = mean(mesaure.values.cc.svm[,2])
subset_cc_mean.svm = mean(mesaure.values.cc.svm[,3])
f1_cc_mean.svm = mean(mesaure.values.cc.svm[,4])
acc_cc_mean.svm = mean(mesaure.values.cc.svm[,5])

hamloss_ns_mean.svm = mean(mesaure.values.ns.svm[,2])
subset_ns_mean.svm = mean(mesaure.values.ns.svm[,3])
f1_ns_mean.svm = mean(mesaure.values.ns.svm[,4])
acc_ns_mean.svm = mean(mesaure.values.ns.svm[,5])


counts <- matrix(0,7,4)

counts[1,1] = hamloss_mean
counts[1,2] = subset_mean
counts[1,3] = f1_mean
counts[1,4] = acc_mean

counts[2,1] = hamloss_cc_mean.tree
counts[2,2] = subset_cc_mean.tree
counts[2,3] = f1_cc_mean.tree
counts[2,4] = acc_cc_mean.tree

counts[3,1] = hamloss_ns_mean.tree
counts[3,2] = subset_ns_mean.tree
counts[3,3] = f1_ns_mean.tree
counts[3,4] = acc_ns_mean.tree

counts[4,1] = hamloss_cc_mean.naiveBayes
counts[4,2] = subset_cc_mean.naiveBayes
counts[4,3] = f1_cc_mean.naiveBayes
counts[4,4] = acc_cc_mean.naiveBayes

counts[5,1] = hamloss_ns_mean.naiveBayes
counts[5,2] = subset_ns_mean.naiveBayes
counts[5,3] = f1_ns_mean.naiveBayes
counts[5,4] = acc_ns_mean.naiveBayes

counts[6,1] = hamloss_cc_mean.svm
counts[6,2] = subset_cc_mean.svm
counts[6,3] = f1_cc_mean.svm
counts[6,4] = acc_cc_mean.svm

counts[7,1] = hamloss_ns_mean.svm
counts[7,2] = subset_ns_mean.svm
counts[7,3] = f1_ns_mean.svm
counts[7,4] = acc_ns_mean.svm

counts

barplot(counts, main="Miary jakosci",
        xlab="Number of Gears", col=c("black","darkgray","lightgray", "darkgreen", "lightgreen","darkblue", "lightblue"),
        names.arg=c("Hamming Loss", "Subset", "F1","Accuracy"), beside=TRUE,
        legend = c("random forest","chain wrapper tree","nested stacking wrapper tree","chain wrapper Naive Bayes","nested stacking wrapper Naive Bayes"
                   ,"chain wrapper SVM","nested stacking wrapper SVM"))


