library("utiml")
library("mlr")
library("BBmisc")
library("mldr")
library("ParamHelpers")

multilabelChain <- function(learner, mydata, binary.label.data) {
  
  reponse_vector <- rep(0, dim(mydata)[2])
  
  # permLabels <- sample(binary.label.data)
  permLabels <- (binary.label.data)
  permLabelsNames <- names(permLabels)
  nr_labels = length(permLabelsNames)
  
  modList <- list()
  
  
  for (i in 1 : nr_labels)
  {
    chain = cbind(mydata, permLabels[i])
    
    
    binarytask = makeClassifTask(id = "BinaryClassification", data = chain, target = permLabelsNames[i], positive = "TRUE")
    mod = train(learner, binarytask)
    task.pred = predict(mod, task = binarytask)
    response = getPredictionResponse(task.pred)
    
    
    for(j in 1:length(response))
    {
      if(response[[j]]==TRUE)
        reponse_vector[j] <- 1
      else
        reponse_vector[j] <- 0
    }
    
    
    
    new_data = cbind(mydata, reponse_vector)
    
    col_name <- permLabelsNames[i]
    colnames(new_data)[dim(new_data)[2]] <- col_name
    
    mydata <- new_data
    
    modList[[i]] <- mod
    class(modList) <- c("myChainClassifier")
    
  }
  
  names(modList) <- permLabelsNames
  
  return(modList)
}



predict.myChainClassier <- function(model,testData,labelData)
{
  predList <- list()
  
  reponse_vector <- rep(0, dim(testData)[1])
  
  labels <- names(model)
  nr_labels <- length(labels)
  
  predMatrix <- matrix(0,dim(testData)[1],nr_labels)
  
  for (i in 1 : nr_labels)
  {
    data.label <- labelData[labels[i]]
    
    chain = cbind(testData, data.label)
    
    predictModel = model[[i]]
    
    binarytask = makeClassifTask(id = "BinaryClassification", data = chain, target = labels[i], positive = "TRUE")
    task.pred = predict(predictModel, task = binarytask)
    response = getPredictionResponse(task.pred)
    
    
    for(j in 1:length(response))
    {
      if(response[[j]]==TRUE)
        reponse_vector[j] <- 1
      else
        reponse_vector[j] <- 0
    }
    
    new_data = cbind(testData, reponse_vector)
    
    col_name <- labels[i]
    colnames(new_data)[dim(new_data)[2]] <- col_name
    
    testData <- new_data
    
    
    
    # predList[[i]] <- reponse_vector
    #class(predList) <- c("myChainClassifier")
    
    predMatrix[,i] <- reponse_vector
    
    
  }
  
  colnames(predMatrix) <- labels
  
  return(predMatrix)
}
