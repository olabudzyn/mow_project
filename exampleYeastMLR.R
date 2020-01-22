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
yeast.task

#stworzenie klasyfikatora
lrn.rfsrc = makeLearner("multilabel.randomForestSRC")
lrn.rfsrc

# podzial danych na testowe i trenujace
n = getTaskSize(yeast.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)

# uczenie => powstaje model
mod = train(lrn.rfsrc, yeast.task, subset = train.set)
mod

# predykcja dla zestawu testowego
pred = predict(mod, task = yeast.task, subset = test.set)
#names(as.data.frame(pred))


listMeasures("multilabel")

# pomiary parametrów
performance(pred, measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))

# resampling
rdesc = makeResampleDesc("Subsample", iters = 10, split = 2/3)
r = resample(lrn.rfsrc, yeast.task, rdesc,measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))
r
