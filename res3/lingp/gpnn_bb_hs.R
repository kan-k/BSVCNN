# R script

# 11 Nov, after 1000 steps, change learning rate to 5e-11 from 5e-10, and after 2000, to 5e-12
#differ from gpnn_gp from the init variance of theta

#2 Feb BB with HS for first-layer weights proir variance

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
p_load(extraDistr)

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))
set.seed(JobId)

print("Starting")

print('############### Test Optimised ###############')


filename <- "feb2_gpnn_bb_hs_beta5"
# prior.var <- 0.05 #was 0.05
learning_rate <- 0.99 #for slow decay starting less than 1
prior.var.bias <- 1
epoch <- 500 #was 500
beta.bb<- 0.5
lr.init <- learning_rate


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
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return((1 - RSS/TSS)*100)
}

#Define ReLU
relu <- function(x) sapply(x, function(z) max(0,z))
relu.prime <- function(x) sapply(x, function(z) 1.0*(z>0))

#Define Mean Squared Error
mse <- function(pred, true){mean((pred-true)^2)}

#Losses
loss.train <- vector(mode = "numeric")
loss.val <- vector(mode = "numeric")
map.train <- vector(mode = "numeric")
#r squared
rsq.train <- vector(mode = "numeric")
rsq.val <- vector(mode = "numeric")
#Predictions
pred.train.ind <- vector(mode = "numeric")
pred.train.val <- vector(mode = "numeric")
pred.test.ind <- vector(mode = "numeric")
pred.test.val <- vector(mode = "numeric")
#Epoch learning rate
lr.vec <- vector(mode = "numeric")
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
# n.expan <- choose(6+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train <-  unlist(ind.temp[1,])
n.train <- length(train.test.ind$train)

# source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp4.R")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_centroids_fixed_100.540.feather"))))


#Length


time.taken <- Sys.time() - start.time
cat("Loading data complete in: ", time.taken)


print("Getting mini batch")
#Get minibatch index 
batch_size <- 500


#NN parameters
it.num <- 1


#Aux variable
xi <- rinvgamma(n.mask,0.5,1)
nu <- matrix(rinvgamma(n.mask*p.dat,0.5,1),nrow=n.mask,ncol=p.dat)   
#
lambda.mat <- matrix(,nrow=n.mask,ncol=p.dat)
for(m in 1:n.mask){
  lambda.mat[m,] <- rinvgamma(p.dat, 0.5,1/nu[m,])
}
tau.vec <- rinvgamma(n.mask, 0.5,1/xi)
prior.var <- tau.vec*lambda.mat

#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- var(age[train.test.ind$train])
y.sigma.vec <- y.sigma

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# weights <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- matrix(, ncol = p.dat, nrow = n.mask)
for(i in 1:n.mask){
  weights[i,] <- rnorm(p.dat,0,sqrt(prior.var*y.sigma))
}

#Initialising bias (to 0)
bias <- rnorm(n.mask)

time.train <-  Sys.time()

#Start epoch
for(e in 1:epoch){
  
  #randomise data
  mini.batch <- batch_split(left = train.test.ind$train, batch_size = batch_size)
  num.batch <- length(mini.batch$train)
  
  grad_x <- 0 #For BB
  
  time.epoch <-  Sys.time()
  #Start batch
  for(b in 1:num.batch){
    
    minibatch.size <- length(mini.batch$train[[b]])
    
    print(paste0("Epoch: ",e, ", batch number: ", b))
    #3 Feed it to next layer
    
    hidden.layer <- apply(t(t(res3.dat[mini.batch$train[[b]], ]  %*% t(weights)) + bias), 2, FUN = relu)
    
    #Hidden layer
    # z.nb <- cbind(1, hidden.layer %*% partial.gp.centroid)
    # hs_fit_SOI <- fast_horseshoe_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    # fit.lm <- lm(age[mini.batch$train[[b]]] ~ hidden.layer) #OLS Regress on the minibatch
    
    z.nb <- cbind(1, hidden.layer %*% partial.gp.centroid)
    hs_fit_SOI <- fast_normal_lm(age[mini.batch$train[[b]]],z.nb) #This also gives the bias term
    beta_fit <- data.frame(HS = partial.gp.centroid %*% hs_fit_SOI$post_mean$betacoef[-1]) #This is the weights of hidden layers with
    l.bias <- hs_fit_SOI$post_mean$betacoef[1]
    
    # beta_fit <- data.frame(HS = partial.gp.centroid %*% hs_fit_SOI$post_mean$betacoef[-1]) #This is the weights of hidden layers with
    #Output layer
    # hs_in.pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer %*%beta_fit$HS
    hs_in.pred_SOI <- l.bias + hidden.layer %*%beta_fit$HS
    loss.train <- c(loss.train, mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]))
    rsq.train <- c(rsq.train, rsqcal(age[mini.batch$train[[b]]],hs_in.pred_SOI))
    
    temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
    map.train <- c(map.train,n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age[mini.batch$train[[b]]]) +n.mask/2*log(y.sigma) +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) )
    
    
    #Validation
    #Layers
    hidden.layer.test <- apply(t(t(res3.dat[train.test.ind$test, ] %*% t(weights)) + bias), 2, FUN = relu)
    
    #Loss calculation
    # hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + hidden.layer.test %*%beta_fit$HS
    hs_pred_SOI <- l.bias + hidden.layer.test %*%beta_fit$HS
    loss.val <- c(loss.val, mse(hs_pred_SOI,age[train.test.ind$test]))
    rsq.val <- c(rsq.val, rsqcal(age[train.test.ind$test],hs_pred_SOI))
    
    
    ##Keeping the last 5 epochs predictions
    if(e >= (epoch-5)){
      pred.train.ind <- c(pred.train.ind,mini.batch$train[[b]]) 
      pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
      pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
      pred.test.val <- c(pred.test.val,hs_pred_SOI) 
    }
    
    if(it.num < epoch*num.batch){
      #Update weight
      
      #4Update the full weights, fit GP against the full weights using HS-prior model to get normally dist thetas
      grad.loss <- age[mini.batch$train[[b]]] - hs_in.pred_SOI
      
      #Update weight
      grad <- array(,dim = c(minibatch.size,dim(weights)))
      for(j in 1:n.mask){ #nrow of weights = n.mask
        grad[,j,] <- -1/y.sigma*c(grad.loss)*beta_fit$HS[j]*c(relu.prime(hidden.layer[,j]))*res3.dat[mini.batch$train[[b]], ]  
      }
      
      #Take batch average
      grad.m <- apply(grad, c(2,3), mean)
      #####
      # print(summary(c(grad.m)))
      #####
      #Update bias
      grad.b <- 1/y.sigma* c(grad.loss)*t(beta_fit$HS * t(apply(hidden.layer, 2, FUN = relu.prime)))
      
      #Take batch average
      grad.b.m <- c(apply(grad.b, c(2), mean))
      
      # Update sigma
      grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(weights/prior.var)^2)+1/(2*y.sigma)*p.dat*n.mask)
      ####Note here of the static equal prior.var
      #Update theta matrix
      weights <- weights*(1-learning_rate*1/(prior.var*y.sigma)) - learning_rate*grad.m * length(train.test.ind$train)
      #Note that updating weights at the end will be missing the last batch of last epoch
      
      #Update bias
      bias <- bias*(1-learning_rate*1/(prior.var.bias)) - learning_rate*c(grad.b.m) * length(train.test.ind$train)
      
      # Update sigma
      y.sigma <- y.sigma - learning_rate*(grad.sigma.m)
      y.sigma.vec <- c(y.sigma.vec,y.sigma)
      
      delta_f <- c(c(weights/(prior.var*y.sigma) + grad.m*n.train),c(bias/prior.var.bias + grad.b.m*(n.train)))
      grad_x <- beta.bb*delta_f + (1-beta.bb)*grad_x
      x.param <- c(c(weights),c(bias))
      
      
      #Update Cv
      for(i in 1:n.mask){
        nu[i,] <- rinvgamma(p.dat,1,1+1/lambda.mat[i,])
        xi[i] <- rinvgamma(n=1,1, 1+1/tau.vec[i]) #Changed from scale to rate ##Changed again, turns out "extraDistr has rinvgamma amd it masks nimble and has diff names
        
        tau.vec[i] <- rinvgamma((length(weights[i,])+1)/2, 1/xi[i] +1/(2*y.sigma)*sum(weights[i,]^2/lambda.mat[i,]))
        lambda.mat[i,] <- rinvgamma(p.dat,1, 1/nu[i,] + weights[i,]^2/(2*tau.vec[i])*y.sigma)
        
        #HSP isn't really a conjugate... the current bit only works for linear regression
        
      }
      prior.var <- tau.vec*lambda.mat
      
    }
    
    it.num <- it.num +1
    
    # invisible(capture.output(ifelse(it.num >=2000, learning_rate <- lr.init*0.001,ifelse(it.num >=1000, learning_rate <- lr.init*0.01, learning_rate <- lr.init) )))
    # if((it.num %% 200) ==0){
    #   learning_rate <- learning_rate*0.1
    # }
    # learning_rate <- learning_rate
    print(paste0("training loss: ",mse(hs_in.pred_SOI,age[mini.batch$train[[b]]])))
    print(paste0("validation loss: ",mse(hs_pred_SOI,age[train.test.ind$test])))
  }
  
  print(paste0("epoch: ",e," out of ",epoch, ", time taken for this epoch: ",Sys.time() -time.epoch))
  print(paste0("sigma^2: ",y.sigma))
  
  if(e %in% c(epoch/2,epoch)){
    salient.mat <- matrix(,nrow=length(train.test.ind$train), ncol = p.dat)
    for(o in train.test.ind$train){
      salient.mat[which(o==train.test.ind$train),]<- c(beta_fit$HS %*% (relu.prime(bias+weights%*%res3.dat[o,])*weights))
    }
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- abs(colMeans(salient.mat))
    gp.mask.hs@datatype = 16
    gp.mask.hs@bitpix = 32
    writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/',filename,'_sal_',JobId,"_epoch_",e))
  }
  
  #BB
  
  if(e >=2){
    diff_x = x.param - prev_x
    diff_grad_x = grad_x - prev_grad_x
    learning_rate <- 1/num.batch*sum(diff_x*diff_x)/abs(sum(diff_x*diff_grad_x))
  }
  prev_x <- x.param
  prev_grad_x <- grad_x
  lr.vec <- c(lr.vec, learning_rate)
}

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

lr.vec <- c(lr.init,lr.vec[-length(lr.vec)]) #Add first learning rate


write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(map.train,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_map_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(weights),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_weights_',"_jobid_",JobId,'.feather'))
write.csv(bias,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_bias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(y.sigma.vec,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_sigma_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(l.bias,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_lbias_',"_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(lr.vec,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_lr_',"_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(partial.gp.centroid %*% hs_fit_SOI$post_mean$betacoef[-1]),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_lweights_',"_jobid_",JobId,'.feather'))

temp.frame <- as.data.frame(rbind(pred.train.ind,pred.train.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$train)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_inpred_',"_jobid_",JobId,'.feather'))
temp.frame <- as.data.frame(rbind(pred.test.ind,pred.test.val))
colnames(temp.frame) <- NULL
colnames(temp.frame) <- 1:ncol(temp.frame)
# temp.frame<- t(tail(t(temp.frame),length(train.test.ind$test)*5))
write_feather(temp.frame,paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_outpred_',"_jobid_",JobId,'.feather'))