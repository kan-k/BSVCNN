# R script
#Adam with regularisation & decoupled weight decay https://arxiv.org/abs/1711.05101
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))
print("Starting")

start.time <- Sys.time()
#1 Split data into mini batches (train and validation)
get_ind_split <- function(num_datpoint, num_test,num_train, batch_size){
  full.ind<-1:num_datpoint
  #test
  test<-sample(x=full.ind,size=num_test,replace=FALSE)
  #train
  left <-sample(x=setdiff(full.ind,test),size=num_train,replace=FALSE)
  mini_batches <- split(left,ceiling(seq_along(left)/batch_size))
  out=list()
  out$test<-test
  out$train<-mini_batches
  return(out)
}

#Define ReLU
relu <- function(x) sapply(x, function(z) max(0,z))
relu.prime <- function(x) sapply(x, function(z) 1.0*(z>0))

#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}

#Losses
loss.train <- vector(mode = "numeric")
loss.val <- vector(mode = "numeric")



print("Loading data")

#Load data and mask and GP 
#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/dat_rearranged.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age
#Length
n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)
#GP
partial.gp <- array(, dim = c(length(res3.mask.reg),p.dat,n.expan))
for(i in res3.mask.reg){
  partial.gp[i,,] <- t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_",i,"_fixed_100.540.feather"))))
}
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_centroids_fixed_100.540.feather"))))

print("Getting mini batch")

time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)

#Get minibatch index 
batch_size <- 500
mini.batch <- get_ind_split(num_datpoint = n.dat, num_test = 2000, num_train = 2000,batch_size = batch_size)
num.batch <- length(mini.batch$train)
#NN parameters
learning_rate <-10^-(JobId)
epoch <- 40


#Hyperparameter
#Define prior variance of theta
prior_var <- 0.9
#NN parameters
learning_rate <-10^-(1)*JobId
epoch <- 10

#Adam 
#hyperparameter
beta1 <- 0.95
beta2 <- 0.95
eps <- 1e-8
#initialisation
m <- matrix(0,nrow=n.mask, ncol= n.expan )
v <- matrix(0,nrow=n.mask, ncol= n.expan )




print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
theta.matrix <- matrix(rnorm(n.mask*n.expan,0,prior_var),nrow=n.mask, ncol= n.expan) #Initialise with Norm(0,0.9)
#1.2 Multiply the partial weights to partial GP and use it as the actual weights of size (p x 1)
#Initialising
weights <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in res3.mask.reg){
  weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
}
num.it <-1

#Start epoch
for(e in 1:epoch){
  
  #randomise data
  mini.batch <- get_ind_split(num_datpoint = n.dat, num_test = 2000, num_train = 2000,batch_size = batch_size)
  num.batch <- length(mini.batch$train)
  
  time.epoch <-  Sys.time()
  #Start batch
  for(b in 1:num.batch){
    
    print(paste0("Epoch: ",e, ", batch number: ", b))
    #3 Feed it to next layer
    
    hidden.layer <- matrix(,nrow=batch_size,ncol = n.mask)
    for(i in 1:n.mask){
      temp.mul <- (res3.dat[mini.batch$train[[b]], ]  %*% weights[i,]) #Will yield a batch_size x 1 
      #Activate by ReLU and save to hidden layer
      hidden.layer[,i] <- relu(temp.mul) #will yield a vector, not matrix
    }
    #Hidden layer
    z.nb <- cbind(1, hidden.layer %*% partial.gp.centroid)
    hs_fit_SOI <- fast_horseshoe_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    beta_fit <- data.frame(HS = partial.gp.centroid %*% hs_fit_SOI$post_mean$betacoef[-1]) #This is the weights of hidden layers with
    #Output layer
    hs_in.pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer %*%beta_fit$HS
    loss.train <- c(loss.train, mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]))
    
    #Validation
    #Layers
    hidden.layer.test <- matrix(,nrow=2000,ncol = n.mask)
    for(i in 1:n.mask){
      temp.mul.test <- (res3.dat[mini.batch$test, ]  %*% weights[i,]) #Will yield a batch_size x 1 
      #Activate by ReLU and save to hidden layer
      hidden.layer.test[,i] <- relu(temp.mul.test)
    }
    #Loss calculation
    hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer.test %*%beta_fit$HS
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[mini.batch$test]))
    
    #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
    grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
    mean.grad.loss <- mean(age[mini.batch$train[[b]]]-hs_in.pred_SOI)
    #grad <- mean.grad.loss* beta_fit$HS * relu.prime() *res3.dat[mini.batch$train[[1]], ]
    # 1/500x1/500x1/
    #OR
    #This should be in the same dim as `theta.matrix`, so for updating w_ij, we require beta_fit_j *relu.prime(i)*input(i) then take avaerge over batch
    grad <- array(,dim = c(batch_size,dim(theta.matrix)))
    for(j in 1:nrow(theta.matrix)){
      grad[,j,] <- -c(grad.loss)*beta_fit$HS[j]*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ] %*% partial.gp[j,,] 
    }
    #Take batch average
    grad.m <- apply(grad, c(2,3), mean)
    # Full gradient with regularisation
    grad.full <- grad.m + 1/(2*prior_var)*theta.matrix
    
    #ADAM param updates
    m <- beta1*m + (1-beta1)*grad.full
    v <- beta2*v + (1-beta2)*grad.full^2
    
    m.hat <- m/(1-beta1^num.it)
    v.hat <- m/(1-beta2^num.it)
    
    #Update theta matrix
    theta.matrix <- theta.matrix - learning_rate* (m.hat/(sqrt(v.hat)+eps) + 1/(2*prior_var*batch_size)*theta.matrix)
    #Note that updating weights at the end will be missing the last batch of last epoch
    
    #Update weight
    for(i in res3.mask.reg){
      weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
    }
    
    num.it <- num.it+1
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[mini.batch$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
}

time.taken <- Sys.time() - start.time
cat("Training complete in: ", time.taken)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/nn_adamW1_loss_",learning_rate,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/nn_adamW1_weights_',learning_rate,'.feather'))
write_feather(as.data.frame(theta.matrix),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/nn_adamW1_theta_',learning_rate,'.feather'))
