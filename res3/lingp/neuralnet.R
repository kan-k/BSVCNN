# R script

# 11 Nov, after 1000 steps, change learning rate to 5e-11 from 5e-10, and after 2000, to 5e-12

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(neuralnet)
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))
set.seed(JobId)

print("Starting")

print('############### Test Optimised ###############')


filename <- "jan25_neuralnet_150k"




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
map.train <- vector(mode = "numeric")

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
print(paste0('Dimension of dat: ',ncol(res3.dat)))
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
n.train <- length(train.test.ind$train)


print("start training")
time.train <-  Sys.time()
model <- neuralnet( age[train.test.ind$train] ~., hidden = p.dat, err.fct = 'sse', linear.output = TRUE, data = res3.dat[train.test.ind$train,])
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

print("Predicting held-out")
predictions <- compute(model, res3.dat[train.test.ind$test,])$net.result

loss.train <- sqrt(mse(unlist(model$net.result),age[train.test.ind$train]))
loss.val <- sqrt(mse(predictions,age[train.test.ind$test]))

print(paste0('training rmse: ', loss.train))
print(paste0('test rmse: ', loss.val))

print('writing results')
write.csv(c(loss.train,loss.val),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_",filename,"_loss_","_jobid_",JobId,".csv"), row.names = FALSE)


