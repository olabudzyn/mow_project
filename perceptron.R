## implementacja perceptronu, ostatecznie nie wykorzystanego do analizy

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
