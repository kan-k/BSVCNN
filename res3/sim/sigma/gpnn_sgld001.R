# R script

#7 sep, make sigma variable


#Change num epochs to 170
#var(age) = 35, perhaps let tau^2 = 1.5?

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

filename <- "oct5_sgld001"
prior.var <- 0.05
gaus.sd <- 0.001
learning_rate <- gaus.sd^2/2
prior.var.bias <- 1
st <- 0.25
sig.lr <- learning_rate
epoch <- 100



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

#Define STGP
stgp <- function(x){sign(x)*max(0,abs(x)-st)}

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
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res3_sub.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data


res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/sim/sub_res3_dat.feather'))

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <- ind.temp[2,]
train.test.ind$train <- ind.temp[1,]
# hs.out<- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_outpred_noscale_",4,".csv"))
hs.out<- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_outpred_4_addednoise.csv"))
# Update with noise added age
hs.in <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_inpred_4_addednoise.csv"))
# hs.in <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_inpred_noscale_",4,".csv"))
age_tab<- as.data.frame(cbind(c(unlist(train.test.ind$test),unlist(train.test.ind$train)),
                              c(unlist(hs.out),unlist(hs.in))))
colnames(age_tab) <- c("id","age")
missing.unused.ind <- setdiff(1:4263,age_tab$id)
age_tab <- as.data.frame(mapply(c,age_tab,as.data.frame(cbind(missing.unused.ind,100000))))
colnames(age_tab) <- c("id","age")
age_tab <- age_tab[order(age_tab$id),]
age <- age_tab$age




n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

#GP
# partial.gp <- array(, dim = c(length(res3.mask.reg),p.dat,n.expan))
# for(i in res3.mask.reg){
#   partial.gp[i,,] <- t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_",i,"_fixed_100.540.feather"))))
# }
source("/well/nichols/users/qcv214/bnn2/res3/sim/nn_v_wb_first_layer_gp.R")
# source("/well/nichols/users/qcv214/bnn2/res3/sim/second_layer_gp.R")

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


#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- tail(as.vector(unlist(read.csv( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_oct5_adjsigmabiasstgpols_sigma__jobid_1.csv'))),1)
y.sigma.vec <- y.sigma

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
theta.matrix <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_oct5_adjsigmabiasstgpols_theta_',"_jobid_",1,'.feather')))


#1.2 Multiply the partial weights to partial GP and use it as the actual weights of size (p x 1)
#Initialising weights
weights <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in res3.mask.reg){
  weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
}
weights <- matrix(sapply(weights,stgp),ncol = p.dat)

#Initialising bias (to 0)
bias<- as.vector(unlist(read.csv( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_oct5_adjsigmabiasstgpols_bias__jobid_1.csv')))

#Gaussian noise
gaus.noise <- matrix(,nrow=n.mask, ncol= n.expan)

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
    for(i in 1:n.mask){
      gaus.noise[i,] <- rnorm(n.expan,0,gaus.sd)
    }
    hidden.layer <- matrix(,nrow=batch_size,ncol = n.mask)
    for(i in 1:n.mask){
      temp.mul <- (res3.dat[mini.batch$train[[b]], ]  %*% weights[i,]) + bias[i] #Will yield a batch_size x 1 + bias of that region
      #Activate by ReLU and save to hidden layer
      hidden.layer[,i] <- relu(temp.mul) #will yield a vector, not matrix
    }
    #Hidden layer
    fit.lm <- lm(age[mini.batch$train[[b]]] ~ hidden.layer) #OLS Regress on the minibatch
    beta_fit <- coefficients(fit.lm)
    beta_fit[is.na(beta_fit)] <- 0
    #Output layer
    hs_in.pred_SOI <- cbind(1,hidden.layer)%*%beta_fit
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
    hs_pred_SOI <- cbind(1,hidden.layer.test)%*%beta_fit
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[train.test.ind$test]))
    pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
    pred.test.val <- c(pred.test.val,hs_pred_SOI) 
    
    
    #Update weight
    
    #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
    grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
    
    #Update weight
    grad <- array(,dim = c(batch_size,dim(theta.matrix)))
    for(j in 1:nrow(theta.matrix)){ #nrow of theta.matrix = n.mask
      grad[,j,] <- -1/y.sigma*c(grad.loss)*beta_fit[j+1]*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ] %*% partial.gp[j,,] 
    }
    #Take batch average
    grad.m <- apply(grad, c(2,3), mean)
    
    #Update bias
    grad.b <- matrix(,nrow = batch_size,ncol = n.mask)
    for(j in 1:n.mask){
      grad.b[,j] <- -1/y.sigma*c(grad.loss)*beta_fit[j+1]*c(relu.prime(hidden.layer[,j]))
    }
    #Take batch average
    grad.b.m <- c(apply(grad.b, c(2), mean))
    
    # Update sigma
    grad.sigma.m <- mean(-length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*prior.var*y.sigma^2)*sum(c(theta.matrix)^2)+1/(2*y.sigma)*n.expan*n.mask)
    ####Note here of the static equal prior.var
    #Update theta matrix
    theta.matrix <- theta.matrix*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train) - gaus.noise
    #Note that updating weights at the end will be missing the last batch of last epoch
    
    #Update bias
    bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train) - rnorm(1,0,gaus.sd)
    
    # Update sigma
    y.sigma <- y.sigma - sig.lr*(grad.sigma.m) - rnorm(1,0,gaus.sd)
    y.sigma.vec <- c(y.sigma.vec,y.sigma)
    
    #Update weight
    for(i in res3.mask.reg){
      weights[i,] <- partial.gp[i,,] %*% theta.matrix[i,]
    }
    weights <- matrix(sapply(weights,stgp),ncol = p.dat)
    it.num <- it.num +1
    # learning_rate <- learning_rate
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
}

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write_feather(as.data.frame(theta.matrix),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_theta_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)

temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_outpred_',"_jobid_",JobId,'.feather'))