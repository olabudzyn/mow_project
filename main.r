install.packages("utiml")
install.packages("RWeka")

library("utiml")
library("rJava")
library("RWeka")
library(mldr)


myyeast <- read.arff("yeast")
myscene <- read.arff("scene")



yeast <- read.arff("yeast")

yeast_train <- read.arff("yeast-train",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")
yeast_test <- read.arff("yeast-test",use_xml=TRUE,auto_extension = TRUE, "yeast.xml")

my_yeast_train <- yeast_train[["dataframe"]]
yeast_train_label <- my_yeast_train[yeast_train[["labelIndices"]]]
yeast_train_attr <- my_yeast_train[1:103]

my_yeast_test <- yeast_test[["dataframe"]]
yeast_test_label <- my_yeast_test[yeast_test[["labelIndices"]]]
yeast_test_attr <- my_yeast_test[1:103]




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



generateChain <- function(attr, label, eta, epoch) {
  weightMatrix <- matrix(0, dim(label)[2], dim(attr)[2])
  
  
  for (i in 1:dim(label)[2])
  {
    perceptron_out <- perceptron(attr, label[, i], eta, epoch)
    
    weights = perceptron_out[[1]]
    
    print(weights)
    weightMatrix[i,] <- weights
    
    r <- results(weights, attr)
    atrr <- cbind(attr, t(r))
  }
  
  
  
  plot(1:epoch,
       perceptron_out[[2]],
       type = "l",
       lwd = 2,
       col = "red",
       xlab = "epoch #",
       ylab = "errors"
  )
  title("B³êdy - liczba epok = , wsp. uczenia = 0.01")
  
  
  return(weightMatrix)
}




chain <- generateChain(yeast_train_attr, yeast_train_label, 0.001, 5)

result <- results(chain[14, ], yeast_test_attr)






# plot data - a picture is worth a 1000 words. Melt data => then ggplot
library(ggplot2)
ggplot(irissubdf, aes(x = sepal, y = petal)) +
  geom_point(aes(colour = species, shape = species), size = 3) +
  xlab("sepal length") +
  ylab("petal length") +
  ggtitle("Species vs sepal and petal lengths")
# add binary labels corresponding to species - Initialize all values to 1
# add setosa label of -1. The binary +1, -1 labels are in the fourth
# column. It is better to create two separate data frames: one containing
# the attributes while the other contains the class values.













