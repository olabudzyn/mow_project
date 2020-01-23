library("mlr")
library("BBmisc")
library("mldr")
library("ParamHelpers")


set.seed(1729)

# wczytanie danych prawidlowe
scene <- read.arff("scene")

data = scene[["dataframe"]]

target = labels(scene[["attributes"]])
labels = target[scene[["labelIndices"]]]

feats = setdiff(target, labels)
new_data = data[feats]
logic_data <- lapply(data[labels], as.logical)

# stworzenie zadania
scene.task = makeMultilabelTask(id = "multi", data = cbind(new_data,logic_data), target = labels)
scene.task

#stworzenie klasyfikatora
lrn.rfsrc = makeLearner("multilabel.randomForestSRC")
lrn.rfsrc

# podzial danych na testowe i trenujace
n = getTaskSize(scene.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)

# uczenie => powstaje model
mod = train(lrn.rfsrc, scene.task, subset = train.set)
mod

# predykcja dla zestawu testowego
pred = predict(mod, task = scene.task, subset = test.set)
#names(as.data.frame(pred))



#stworzenie klasyfikatora binarnego
binary.learner = makeLearner("classif.rpart")


###### Chains Wrapper
lrn.cc = makeMultilabelClassifierChainsWrapper(binary.learner)

# uczenie => powstaje model dla chain wrapper
mod_cc = train(lrn.cc, scene.task, subset = train.set)
# predykcja dla zestawu testowego
pred_cc = predict(mod_cc, task = scene.task, subset = test.set)




###### Nested Stacking Wrapper
lrn.ns = makeMultilabelNestedStackingWrapper(binary.learner, order = NULL, cv.folds = 2)

# uczenie => powstaje model dla nested stacking
mod_ns = train(lrn.ns, scene.task, subset = train.set)
# predykcja dla zestawu testowego
pred_ns = predict(mod_ns, task = scene.task, subset = test.set)



listMeasures("multilabel")

# pomiary parametrów
performance(pred, measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))
performance(pred_cc, measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))
performance(pred_ns, measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))


# resampling
rdesc = makeResampleDesc("Subsample", iters = 10, split = 2/3)
r = resample(lrn.rfsrc, scene.task, rdesc,measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))
r_cc = resample(lrn.cc, scene.task, rdesc,measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))
r_ns = resample(lrn.ns, scene.task, rdesc,measures = list(multilabel.hamloss, multilabel.subset01, multilabel.f1, multilabel.acc))


mes_values <- r[["measures.test"]]
mes_cc_values <- r_cc[["measures.test"]]
mes_ns_values <- r_ns[["measures.test"]]

hamloss_mean = mean(mes_values[,2])
subset_mean = mean(mes_values[,3])
f1_mean = mean(mes_values[,4])
acc_mean = mean(mes_values[,5])

hamloss_cc_mean = mean(mes_cc_values[,2])
subset_cc_mean = mean(mes_cc_values[,3])
f1_cc_mean = mean(mes_cc_values[,4])
acc_cc_mean = mean(mes_cc_values[,5])

hamloss_ns_mean = mean(mes_ns_values[,2])
subset_ns_mean = mean(mes_ns_values[,3])
f1_ns_mean = mean(mes_ns_values[,4])
acc_ns_mean = mean(mes_ns_values[,5])

counts <- matrix(0,3,4)


counts[1,1] = hamloss_mean
counts[1,2] = subset_mean
counts[1,3] = f1_mean
counts[1,4] = acc_mean

counts[2,1] = hamloss_cc_mean
counts[2,2] = subset_cc_mean
counts[2,3] = f1_cc_mean
counts[2,4] = acc_cc_mean

counts[3,1] = hamloss_ns_mean
counts[3,2] = subset_ns_mean
counts[3,3] = f1_ns_mean
counts[3,4] = acc_ns_mean


counts


barplot(counts, main="Miary jakoœci",
        xlab="Number of Gears", col=c("darkblue","red","black"),
        names.arg=c("Hamming Loss", "Subset", "F1","Accuracy"), beside=TRUE,
        legend = c("random forest","chain wrapper","nested stacking wrapper"))


