# R script

#IG alpha = 5, and use mean for initialisation
#With this, expectation is 0.1

## Increase num epoch to 200

#SGD with 12 x 12 GP


if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(nimble)

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))
print("Starting")

filename <- "sep7_nnig"
learning_rate <- 0.1 #for slow decay starting less than 1
epoch <- 80



start.time <- Sys.time()
#1 Split data into mini batches (train and validation)

train_test_split<- function(num_datpoint, num_test,num_train){
  full.ind<-1:num_datpoint
  #test
  test<-sample(x=full.ind,size=num_test,replace=FALSE)
  #train
  left <-sample(x=setdiff(full.ind,test),size=num_train,replace=FALSE)
  mini_batches <- split(left,ceiling(seq_along(left)/batch_size))
  out=list()
  out$test<-test
  out$train<-left
  return(out)
}

batch_split <- function(left, batch_size){
  left <- sample(x=left,size=length(left),replace=FALSE)
  mini_batches <- split(left,ceiling(seq_along(left)/batch_size))
  out=list()
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
#Predictions
pred.train.ind <- vector(mode = "numeric")
pred.train.val <- vector(mode = "numeric")
pred.test.ind <- vector(mode = "numeric")
pred.test.val <- vector(mode = "numeric")

print("Loading data")

#Load data and mask and GP 
#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age_tab <- age_tab[order(age_tab$id),]
age <- age_tab$age

n.mask <- length(res3.mask.reg)
n.expan <- choose(6+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)


ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train <-  unlist(ind.temp[1,])



source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_centroids_fixed_100.540.feather"))))

#Length


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500
# seed <- prior.var.mat[JobId,1]
# set.seed(seed)
# train.test.ind <- train_test_split(num_datpoint = n.dat, num_test = 2000, num_train = 2000)
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <- unlist(ind.temp[2,])
train.test.ind$train <- unlist(ind.temp[1,])

#NN parameters
it.num <- 1


#Initial parameters for inverse gamma
alpha.init <- rep(11,n.mask) #shape
beta.init <- rep(0.5,n.mask) #scale


#Storing inv gamma
conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
# conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)

#Define init var
prior.var <- beta.init/(alpha.init-1) #Mean of IG
y.sigma <- var(age[train.test.ind$train])
# y.sigma.vec <- y.sigma
# C2 <- 1/(prior_var)

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
for(i in 1:n.mask){
  theta.matrix[i,] <- rnorm(n.expan,0,sqrt(0.1))
}

#1.2 Multiply the partial weights to partial GP and use it as the actual weights of size (p x 1)
#Initialising weights
weights <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in res3.mask.reg){
  weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
}
#Initialising bias (to 0)
bias <- rep(0,n.mask)


time.train <-  Sys.time()

#Start epoch
for(e in 1:epoch){
  
  #randomise data
  mini.batch <- batch_split(left = train.test.ind$train, batch_size = batch_size)
  num.batch <- length(mini.batch$train)
  
  time.epoch <-  Sys.time()
  #Start batch
  for(b in 1:num.batch){
    
    print(paste0("Epoch: ",e, ", batch number: ", b))
    #3 Feed it to next layer
    
    hidden.layer <- matrix(,nrow=batch_size,ncol = n.mask)
    for(i in 1:n.mask){
      temp.mul <- (res3.dat[mini.batch$train[[b]], ]  %*% weights[i,]) + bias[i] #Will yield a batch_size x 1 + bias of that region
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
    pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
    pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
    
    #Validation
    #Layers
    hidden.layer.test <- matrix(,nrow=2000,ncol = n.mask)
    for(i in 1:n.mask){
      temp.mul.test <- (res3.dat[train.test.ind$test, ]  %*% weights[i,]) + bias[i] #Will yield a batch_size x 1 + bias of that region
      #Activate by ReLU and save to hidden layer
      hidden.layer.test[,i] <- relu(temp.mul.test)
    }
    #Loss calculation
    hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer.test %*%beta_fit$HS
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[train.test.ind$test]))
    pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
    pred.test.val <- c(pred.test.val,hs_pred_SOI)
    
    
    #Update weight
    
    #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
    grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
    #grad <- mean.grad.loss* beta_fit$HS * relu.prime() *res3.dat[mini.batch$train[[1]], ]
    # 1/500x1/500x1/
    #OR
    #This should be in the same dim as `theta.matrix`, so for updating w_ij, we require beta_fit_j *relu.prime(i)*input(i) then take avaerge over batch
    
    #Update weight
    grad <- array(,dim = c(batch_size,dim(theta.matrix)))
    for(j in 1:nrow(theta.matrix)){ #nrow of theta.matrix = n.mask
      grad[,j,] <- -1/y.sigma*c(grad.loss)*beta_fit$HS[j]*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ] %*% partial.gp[j,,] 
    }
    #Take batch average
    grad.m <- apply(grad, c(2,3), mean)
    
    #Update bias
    grad.b <- matrix(,nrow = batch_size,ncol = n.mask)
    for(j in 1:n.mask){
      grad.b[,j] <- -1/y.sigma*c(grad.loss)*beta_fit$HS[j]*c(relu.prime(hidden.layer[,j]))
    }
    #Take batch average
    grad.b.m <- c(apply(grad.b, c(2), mean))
    
    #Update sigma
    # grad.sigma.m <- mean(1/y.sigma - 1/(2*y.sigma^2)*c(grad.loss)^2-1/(2*prior.var*y.sigma^2)*sum(c(theta.matrix)^2))
    
    #Update theta matrix
    theta.matrix <- theta.matrix*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m
    #Note that updating weights at the end will be missing the last batch of last epoch
    
    #Update bias
    bias <- bias - learning_rate*c(grad.b.m)
    
    #Update sigma
    # y.sigma.old <- y.sigma 
    # y.sigma <- y.sigma - learning_rate*(grad.sigma.m)
    # y.sigma.vec <- c(y.sigma.vec,y.sigma)
    
    #Update weight
    for(i in res3.mask.reg){
      weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
    }
    
    #Update Cv
    
    for(i in 1:n.mask){
      alpha.shape <- alpha.init[i] + length(theta.matrix[i,])/2
      # alpha.shape <- alpha.init[i] # Keep alpha the same
      beta.scale <- beta.init[i] + sum(theta.matrix[i,]^2)/(2*y.sigma)
      prior.var[i] <- rinvgamma(n = 1, alpha.shape, scale = beta.scale)
      
      conj.alpha[i,it.num] <- alpha.shape
      conj.beta[i,it.num] <- beta.scale
      conj.invgamma[i,it.num] <- prior.var[i]
    }
    
    # C2 <- 1/(2*prior.var)
    
    # conj.cv[,it.num] <- C2
    
    it.num <- it.num +1
    # learning_rate <- 0.01
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  # print(paste0("sigma^2: ",y.sigma))
}

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write_feather(as.data.frame(theta.matrix),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_theta_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)

temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))
#inv gamme param
write.csv(conj.alpha,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_alpha_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.beta,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_beta_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(conj.invgamma,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_invgam_',"_jobid_",JobId,".csv"), row.names = FALSE)