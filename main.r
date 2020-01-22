install.packages("utiml")
install.packages("RWeka")

library("utiml")
library("rJava")
library("RWeka")
library(mldr)
## Loading required package: mldr
## Loading required package: parallel
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## The following object is masked from 'package:stats':
## 
##     lowess


myyeast <- read.arff("yeast")

myscene <- read.arff("scene")

hemydata_mldr <- mldr("yeast", use_xml = TRUE, auto_extension = TRUE, "yeast.xml")

mydata_part <- create_holdout_partition(mydata_mldr, c(train=0.65, test=0.35), "iterative")

head(myscene)

d = getTaskData(myyeast.task)


# plot data - a picture is worth a 1000 words. Melt data => then ggplot
library(ggplot2)
ggplot(irissubdf, aes(x = sepal, y = petal)) + 
  geom_point(aes(colour=species, shape=species), size = 3) +
  xlab("sepal length") + 
  ylab("petal length") + 
  ggtitle("Species vs sepal and petal lengths")
# add binary labels corresponding to species - Initialize all values to 1
# add setosa label of -1. The binary +1, -1 labels are in the fourth  
# column. It is better to create two separate data frames: one containing
# the attributes while the other contains the class values.
irissubdf[, 4] <- 1
irissubdf[irissubdf[, 3] == "setosa", 4] <- -1






data<-read.arff("yeast")
mydata <-data[["dataframe"]]

label <-mydata[data[["labelIndices"]]]
attr <- mydata[1:103]



generateChain <- function(attr,label,eta,epoch) {
  
  
  weightMatrix <- matrix(0, dim(label)[2],dim(attr)[2])
  

  for(i in 1:dim(label)[2] )  
  {
   weights <- perceptron(attr,label[,i], eta, epoch) 

   weightMatrix[i, ] <- weights
   
    r <- results(weights,attr)                 
    
    atrr <- cbind(attr,t(r))
  }
  
  
  return(weightMatrix)

}


chain <- generateChain(attr,label,0.001,5)  

result <- results(chain[14,],attr)  

perceptron <- function(x, y, eta, niter) {
  
  # initialize weight vector
 ## weight <- rep(0, dim(x)[2]+1)
  
  errors <- rep(0, niter)
  weight <- t(runif(dim(x)[2], min=0,max=1))
  

  # loop over number of epochs niter
  for (jj in 1:niter) {
    
    # loop through training data set
    for (ii in 1:length(y)) {
      
      # Predict binary label using Heaviside activation 
      # function
      
    ##  z <- sum(weight[2:length(weight)] * 
    ##             as.numeric(x[ii, ])) + weight[1]
      
      
    ##  if(z < 0) {
    ##    ypred <- -1
    ##  } else {
    ##    ypred <- 1
    ##  }
      
      beta=5;


     z <-  as.numeric(x[ii,])%*% t(weight)

    ypred = 1 / ( 1 + exp ( -beta * z ) ); 

      

    
      a <- y[ii] - ypred
   

      weightdiff <- eta * (y[ii] - ypred) *   as.numeric(x[ii, ])
    
      weight <- weight + t(weightdiff)
   
  
  ##     Update error function
  ##    if ((y[ii] - ypred) != 0.0) {
  ##      errors[jj] <- errors[jj] + 1
  ##    }
      
    }
  }
  
  

  # weight to decide between the two species 
 
##print(weight)
print(weight)
  return(weight)


}


err <- perceptron(data_attributes,label[,1], 0.001, 50)



results <- function(weights,x) {
  val <- rep(0, dim(x)[1])
  
  for(ii in 1:dim(x)[1]){
    
    beta = 5
    z <-  as.numeric(x[ii,])%*% t(weight)
    
    ypred = 1 / ( 1 + exp ( -beta * z ) ); 
    
    
    
    
    if(abs(ypred>0.5))
      val[ii] = 1
    else
      val[ii] = 0
       
    
  }
    return(val)
}















plot(1:10, err, type="l", lwd=2, col="red", xlab="epoch #", ylab="errors")
title("Errors vs epoch - learning rate eta = 0.01")

label[label== 0] <- -1
