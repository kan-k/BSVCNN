# R script

# This is taken `mala_weights_q_fixed.R`
#Explore lr s.t. 
#Reduce number of samples from 2,000 to 200
#24 mar, I forgot to record step.size.vec, added again
#30 add recording of out of sample predictions to 1000

#6 AUG
#ADDED BIAS, AT FIRST ADDED HIDDEN LAYER BUT I'M DELETING SINCE I DON'T HAVE GRADIENTS
#NEED TO CHECK GRADIENT OF SIGMA, IT DOESN'T CONTAIN BIAS


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
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)
set.seed(JobId)

print("Starting")

print('############### Test Optimised ###############')

step_size <- 31e-7 ##Arbitrary
adap.step.size <- 1
acc.want <- 0.574

filename <- "aug7_mala_weights_sub_adapt1_diffinit_fixed"
init.num <- JobId 
prior.var <- 0.05 #was 0.05
start.b <- 1e1 #Originally 1e9
start.a <- 1e-12
start.gamma <- 0.55
learning_rate <- start.a*(start.b+1)^(-start.gamma) #for slow decay starting less than 1 #
prior.var.bias <- 1
epoch <- 300 #was 500
# burnin.epoch <- 100 #Epochs until SGLD kicks in
record.epoch <- 200 #Record last 100 epochs
# beta.bb<- 0.5
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
phi <- function(k){(k+1)}

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
pre.lr.vec <- vector(mode = "numeric")
prod.lr.vec <-  vector(mode = "numeric")
ck.old <- 1

#step_size
step.size.vec <- step_size


#Recaord noise of gradients
#####Need to change in other cases
grad.noise <- matrix(nrow = 12*2,ncol=epoch)

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
train.test.ind$train <-  unlist(ind.temp[1,])[1:200]
n.train <- length(train.test.ind$train)


#Filter only data from training:
res3.dat.test <- res3.dat[train.test.ind$test,]
age.test <- age[train.test.ind$test]

res3.dat <- res3.dat[train.test.ind$train,]
age <- age[train.test.ind$train]


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

#Initial parameters for inverse gamma
alpha.init <-  read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_alpha__jobid_",init.num,".csv")) #shape
alpha.init <- as.numeric(c(alpha.init[,length(alpha.init)-1]))
beta.init <-  read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_beta__jobid_",init.num,".csv")) #scale
beta.init <- as.numeric(c(beta.init[,length(beta.init)-1]))

#Storing inv gamma
conj.alpha <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.beta <- matrix(, nrow=n.mask,ncol=epoch*4)
conj.invgamma <-matrix(, nrow=n.mask,ncol=epoch*4)
# conj.cv <- matrix(, nrow=n.mask,ncol=epoch*4)

#Define init var
prior.var <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_invgam__jobid_",init.num,".csv"))#Mean of IG
prior.var <- as.numeric(c(prior.var[,length(prior.var)-1]))
print(prior.var)
#Fix prior var to be 0.1
# prior.var <- 1.5
y.sigma <- tail(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_sigma__jobid_",init.num,".csv"))$x,1)
y.sigma.vec <- y.sigma

gaus.sd <- 0

print("Initialisation")
#1 Initialisation
#1.1 Initialise the partial weights around normal dist as a matrix of size (nrow(bases..ie choose...) x number of neurons in 2nd layer ie#regions)
# theta.matrix <- matrix(,nrow=n.mask, ncol= n.expan)
weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_weights__jobid_',init.num,'.feather')))

#Initialising bias (to 0)
bias <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_bias__jobid_",init.num,".csv"))$x

##Load Hidden Layer
#12 x 1
l.weights <- as.matrix(read_feather(paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_lweights__jobid_',init.num,'.feather')))
beta_fit <- data.frame(HS = l.weights)
colnames(beta_fit) <- "HS"

# 1 x 1
l.bias <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_feb27_gpnn_bb_ig_init_ext_bbs_test_lbias__jobid_",init.num,".csv"))$x


time.train <-  Sys.time()

print("Finished loading bias and weights")
#######################################################################################################################################

##### This is wrong, it needs to incorporate everything!!!!!


# Set the number of iterations and burn-in period
num_iterations <- 2000
# burn_in <- 0

# Initialize the accepted parameter values
accepted_params <- matrix(NA, nrow = num_iterations, ncol = 1000) #p.dat*n.mask)
accepted_ratio <- vector(mode='numeric')

# Initialize the acceptance counter
accept_counter <- 0
print("Starting iter")
# Iterate through the algorithm
for (i in 1:num_iterations) {
  time.epoch <-  Sys.time()
  # Compute the log likelihood and log prior
  # Return the sum of the log likelihood and log prior
  
  ###For old, get log posterior and gradients
  ##Cal log pos
  hidden.layer <- apply(t(t(res3.dat%*%t(weights)) + bias), 2, FUN = relu)
  z.nb <- cbind(1, hidden.layer %*% partial.gp.centroid)
  hs_fit_SOI <- fast_normal_lm(age,z.nb,mcmc_sample =1) #This also gives the bias term
  beta_fit <- data.frame(HS = partial.gp.centroid %*% hs_fit_SOI$post_mean$betacoef[-1]) #This is the weights of hidden layers with
  l.bias <- hs_fit_SOI$post_mean$betacoef[1]
  hs_in.pred_SOI <- l.bias + hidden.layer %*%beta_fit$HS
  # loss.train <- c(loss.train, mse(hs_in.pred_SOI,age))
  # rsq.train <- c(rsq.train, rsqcal(age,hs_in.pred_SOI))
  temp.sum.sum.sq <- apply(weights, 1, FUN = function(x) sum(x^2))
  
  #######ADD BIAS
  
  grad.loss <- age - hs_in.pred_SOI
  #I think this is the loglik
 # log.likelihood <- sum(-(n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age)))
  log.likelihood <- (-(n.train/2*log(y.sigma) +1/(2*y.sigma)*n.train*mse(hs_in.pred_SOI,age))) #previously there was *sum *over the whole thing but the original dimension should be 1 so sum shouldn't matter?

  #log.prior <- -(n.mask/2*log(y.sigma) +1/(2*y.sigma)*1/1*sum(beta_fit$HS^2)+1/2*l.bias^2 #hidden layer.  ##########THING IS I DIDN'T SAVE HIDDEN LAYER. Do i need it? Also should I add sigma? Yes and Yes
  #               +n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) ) #final layer 
  log.prior <- -(n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias)^2) ) #final layer 
  #Need and loglik prior for sigma^2??? ==> I think sigma^2 has flat prior, so no need.
  
  log_pos <- log.likelihood + log.prior
  ##Cal log grad
  grad <- array(,dim = c(n.train,dim(weights)))
  for(j in 1:n.mask){ #nrow of theta.matrix = n.mask
    grad[,j,] <- 1/y.sigma*c(grad.loss)*beta_fit$HS[j]*c(relu.prime(hidden.layer[,j]))*res3.dat   #I removed the negative sign 
  } 
  #Take batch average
  grad.m <- apply(grad, c(2,3), mean)
  gradient <- c((weights/(prior.var*y.sigma) + grad.m*n.train)) #Remove the negative signs in front of weights
  
  #BIAS
  grad.b <- 1/y.sigma* c(grad.loss)*t(beta_fit$HS * t(apply(hidden.layer, 2, FUN = relu.prime)))
  grad.b.m <- c(apply(grad.b, c(2), mean))
  gradient.b <- bias/prior.var.bias + grad.b.m*n.train
  
  #Gradient of sigma #############W WHERE IS BIAS HERE?
  grad.sigma.m <- mean(length(train.test.ind$train)/(2*y.sigma) - length(train.test.ind$train)/(2*y.sigma^2)*c(grad.loss)^2-1/(2*y.sigma^2)*sum(c(weights/prior.var)^2)+1/(2*y.sigma)*p.dat*n.mask)
  #y.sigma <- y.sigma - learning_rate*(grad.sigma.m) - rnorm(1,0,gaus.sd)
  
  #proposal
  mean_proposal <- weights + step_size * gradient
  mean_proposal_bias <- bias + step_size*gradient.b
  mean_proposal_y.sigma <- y.sigma + step_size*grad.sigma.m
  
  new_state.flatten <- c(mean_proposal) + sqrt(2*step_size)*rnorm(length(c(mean_proposal)))
  
  #######ADD BIAS
  
  weights_proposal <- matrix(new_state.flatten,ncol = ncol(weights))
  bias_proposal <-mean_proposal_bias + sqrt(2*step_size)*rnorm(length(c(mean_proposal_bias)))
  y.sigma_proposal <- mean_proposal_y.sigma + sqrt(2*step_size)*rnorm(length(c(mean_proposal_y.sigma)))
  
  
  ###For new, get log posterior and gradients
  ##Cal log pos
  hidden.layer_proposal <- apply(t(t(res3.dat%*%t(weights_proposal)) + bias_proposal), 2, FUN = relu)
  z.nb_proposal <- cbind(1, hidden.layer_proposal %*% partial.gp.centroid)
  hs_fit_SOI_proposal <- fast_normal_lm(age,z.nb_proposal,mcmc_sample =1) #This also gives the bias term
  beta_fit_proposal <- data.frame(HS = partial.gp.centroid %*% hs_fit_SOI_proposal$post_mean$betacoef[-1]) #This is the weights of hidden layers with
  l.bias_proposal <- hs_fit_SOI_proposal$post_mean$betacoef[1]
  hs_in.pred_SOI_proposal <- l.bias_proposal + hidden.layer_proposal %*%beta_fit_proposal$HS
  # loss.train <- c(loss.train, mse(hs_in.pred_SOI,age))
  # rsq.train <- c(rsq.train, rsqcal(age,hs_in.pred_SOI))
  temp.sum.sum.sq_proposal <- apply(weights_proposal, 1, FUN = function(x) sum(x^2))
  
  grad.loss_proposal <- age - hs_in.pred_SOI_proposal
  #log.likelihood_proposal <- sum(-(n.train/2*log(y.sigma_proposal) +1/(2*y.sigma_proposal)*n.train*mse(hs_in.pred_SOI_proposal,age) +n.mask/2*log(y.sigma_proposal) +n.mask*p.dat/2*log(y.sigma_proposal)))
  #log.prior_proposal <- -(n.mask*p.dat/2*log(y.sigma) + 1/(2*y.sigma)*sum(1/prior.var*(temp.sum.sum.sq_proposal)))
  
  log.likelihood_proposal <- (-(n.train/2*log(y.sigma_proposal) +1/(2*y.sigma_proposal)*n.train*mse(hs_in.pred_SOI,age))) #previously there was *sum *over the whole thing but the original dimension should be 1 so sum shouldn't matter?
  
  #log.prior_proposal <- -(n.mask/2*log(y.sigma_proposal) +1/(2*y.sigma_proposal)*1/1*sum(beta_fit_proposal$HS^2)+1/2*l.bias_proposal^2 #hidden layer.  ##########THING IS I DIDN'T SAVE HIDDEN LAYER. Do i need it? Also should I add sigma? Yes and Yes
  #               +n.mask*p.dat/2*log(y.sigma_proposal) + 1/(2*y.sigma_proposal)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias_proposal)^2) ) #final layer 
  log.prior_proposal <- -(n.mask*p.dat/2*log(y.sigma_proposal) + 1/(2*y.sigma_proposal)*sum(1/prior.var*(temp.sum.sum.sq))  +1/2*sum(c(bias_proposal)^2) ) #final layer 
  
  log_pos_proposal <- log.likelihood_proposal + log.prior_proposal
  ##Cal log grad
  grad_proposal <- array(,dim = c(n.train,dim(weights_proposal)))
  
  #######ADD BIAS
  
  for(j in 1:n.mask){ #nrow of theta.matrix = n.mask
    grad_proposal[,j,] <- 1/y.sigma*c(grad.loss_proposal)*beta_fit_proposal$HS[j]*c(relu.prime(hidden.layer_proposal[,j]))*res3.dat   #I removed the negative sign 
  } 
  #Take batch average
  grad.m_proposal <- apply(grad_proposal, c(2,3), mean)
  gradient_proposal <- c((weights_proposal/(prior.var*y.sigma) + grad.m_proposal*n.train)) #Remove the negative signs in front of weights
  
  grad.b_proposal <- 1/y.sigma_proposal* c(grad.loss_proposal)*t(beta_fit_proposal$HS * t(apply(hidden.layer_proposal, 2, FUN = relu.prime)))
  grad.b.m_proposal <- c(apply(grad.b_proposal, c(2), mean))
  gradient.b_proposal <- bias_proposal/prior.var.bias + grad.b.m_proposal*n.train
  
  #Gradient of sigma
  grad.sigma.m_proposal <- mean(length(train.test.ind$train)/(2*y.sigma_proposal) - length(train.test.ind$train)/(2*y.sigma_proposal^2)*c(grad.loss_proposal)^2-1/(2*y.sigma_proposal^2)*sum(c(weights_proposal/prior.var)^2)+1/(2*y.sigma_proposal)*p.dat*n.mask)
  #y.sigma <- y.sigma - learning_rate*(grad.sigma.m) - rnorm(1,0,gaus.sd)

  ##########
  # Compute the Metropolis-Hastings acceptance ratio
  
  # log_transition_ratio <- -1/(4*step_size)*sum((c(weights)-c(weights_proposal)-step_size*c(gradient_proposal))^2) +1/(4*step_size)*sum((c(weights_proposal)-c(weights)-step_size*c(gradient))^2)
  
  log_transition_ratio <- -1/(4*step_size)*sum((c(c(weights),bias,y.sigma)-c(c(weights_proposal),bias_proposal,y.sigma_proposal)-step_size*c(c(gradient_proposal),gradient.b_proposal,grad.sigma.m_proposal))^2) +1/(4*step_size)*sum((c(c(weights_proposal),bias_proposal,y.sigma_proposal)-c(c(weights),bias,y.sigma)-step_size*c(c(gradient_proposal),gradient.b_proposal,grad.sigma.m_proposal))^2)
  

  
  log.cur <- log_pos
  log.prop <-  log_pos_proposal
  log_posterior_ratio <- log.prop - log.cur
  
  # print(paste0("proportion overlap old/new: ", round(sum(c(weights_current)==c(weights_proposal))/length(c(weights_current)),4)))
  # print(paste0("log posterior Current: ", log.cur))
  print(paste0("log posterior Proposal: ", log.prop))
  
  log_accept_ratio <- log_posterior_ratio + log_transition_ratio
  
  # print(paste0("ratio of transition prob: ", exp(log_transition_ratio)))
  print(paste0("ratio of posterior prob: ", exp(log_posterior_ratio)))
  accept_ratio <- exp(log_accept_ratio)
  accepted_ratio <- c(accepted_ratio,accept_ratio)
  print(paste0("Accept ratio: ", accept_ratio))
  # Generate a uniform random number
  u <- runif(1)
  # Accept or reject the proposal
  if (u < accept_ratio) {
    weights <- weights_proposal
    bias <- bias_proposal
    y.sigma <- y.sigma_proposal
    accept_counter <- accept_counter + 1
    print("accepted")
    
    ##Recording pred and loss
    
    loss.train <- c(loss.train, mse(hs_in.pred_SOI_proposal,age))
    rsq.train <- c(rsq.train, rsqcal(age,hs_in.pred_SOI_proposal))
    
    hidden.layer.test <- apply(t(t(res3.dat.test %*% t(weights_proposal)) + bias_proposal), 2, FUN = relu)
    hs_pred_SOI <- l.bias_proposal + hidden.layer.test %*%beta_fit_proposal$HS
    loss.val <- c(loss.val, mse(hs_pred_SOI,age.test))
    rsq.val <- c(rsq.val, rsqcal(age.test,hs_pred_SOI))
    
    pred.train.ind <- c(pred.train.ind,train.test.ind$train) 
    pred.train.val <- c(pred.train.val,hs_in.pred_SOI_proposal)
    pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
    pred.test.val <- c(pred.test.val,hs_pred_SOI)    
    
  }else{ #Not accepted, use old param
    ##Recording pred  and loss
    
    loss.train <- c(loss.train, mse(hs_in.pred_SOI,age))
    rsq.train <- c(rsq.train, rsqcal(age,hs_in.pred_SOI))
    
    hidden.layer.test <- apply(t(t(res3.dat.test %*% t(weights)) + bias), 2, FUN = relu)
    hs_pred_SOI <- l.bias + hidden.layer.test %*%beta_fit$HS
    loss.val <- c(loss.val, mse(hs_pred_SOI,age.test))
    rsq.val <- c(rsq.val, rsqcal(age.test,hs_pred_SOI))
    
    pred.train.ind <- c(pred.train.ind,train.test.ind$train) 
    pred.train.val <- c(pred.train.val,hs_in.pred_SOI)
    pred.test.ind <- c(pred.test.ind,train.test.ind$test) 
    pred.test.val <- c(pred.test.val,hs_pred_SOI) 
    
  }
  # Store the accepted parameter values
  accepted_params[i, ] <- c(weights)[1:1000]
  # if (i > burn_in) {
  #   accepted_params[i - burn_in, ] <- c(weights_current)
  # }
  
  new.step.size <- exp( log(step_size)+adap.step.size*(accept_ratio-acc.want))
  step.size.vec <- c(step.size.vec,new.step.size)
  
  prop.change.step.size <- round((1-new.step.size/step_size)*100,2)
  step_size <- new.step.size
  
  ###Tracking performance
  
  print(paste0("Iteration: ",i," out of ",num_iterations,",uncorrected training loss: ",mse(hs_in.pred_SOI,age),", Change lr: ",prop.change.step.size,"%", ", iteration: ",Sys.time() -time.epoch))
  
}

# Print the acceptance rate
accept_rate <- accept_counter / num_iterations
print(paste0("Acceptance rate:", accept_rate))

time.taken <- Sys.time() - start.time
cat("Training complete in: ", time.taken)

# Plot the histogram of the accepted parameter values
# hist(accepted_params[, 1], breaks = 30)

write.csv(rbind(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(rbind(rsq.train,rsq.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_rsq_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(accepted_ratio,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_acceptedratio_","_jobid_",JobId,".csv"), row.names = FALSE)
write.csv(step.size.vec,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_lr_","_jobid_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(accepted_params),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/re_',filename,'_acceptedweights_',"_jobid_",JobId,'.feather'))

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
