# R script
#devtools::install_github("kangjian2016/PMS", force = TRUE) 
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
##3 Dec with white matter, stem removed and thresholded

print("stage 1")
JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

dat_allmat <- read_feather('/well/nichols/users/qcv214/bnn2/smooth/dat_sm.feather')
dat_allmat<- as.matrix(dat_allmat)

age_tab<-as.data.frame(read.csv('/well/nichols/users/qcv214/bnn2/smooth/age_tab.csv'))
colnames(age_tab)[1:2]<-c('id','age')


print("stage 2")

#func
get_ind <- function(num.train,num.test = 1){
  full.ind<-1:dim(dat_allmat)[1]
  train<-sample(x = full.ind,size = num.train,replace = FALSE)
  test<-sample(x=setdiff(full.ind,train),size=num.test,replace=FALSE)
  out=list()
  out$train<-train
  out$test<-test
  return(out)
}
rsqcal2<-function(old,new,ind.old,ind.new){
  ridge_pred<-old
  ridge_pred_new<-new
  no<-length(old)
  nn<-length(new)
  #insample
  y<-age_tab$age[ind.old]
  sserr_ridge<-sum((y-ridge_pred)^2)
  sstot<-var(y)*length(y)
  #outsample
  y_new<-age_tab$age[ind.new]
  sstot_new<-sum((y_new-mean(y))^2)
  sserr_ridge_new<-sum((y_new-ridge_pred_new)^2)
  #Output
  #print(paste0('In-sameple Variance of prediction explained: ',round(1-sserr_ridge/sstot,5)*100,' || RMSE: ', round(sqrt(mean((y-ridge_pred)^2)),4)))
  #print(paste0('Out-sample Variance of prediction explained: ',round(1-sserr_ridge_new/sstot_new,5)*100,' || RMSE: ', round(sqrt(mean((y_new-ridge_pred_new)^2)),4)))
  print('Done')
  out=list()
  out$inrsq<-round(1-sserr_ridge/sstot,5)*100
  out$inmae<-round(median(abs(y-ridge_pred)),4)
  out$outrsq<-round(1-sserr_ridge_new/sstot_new,5)*100
  out$outmae<-round(median(abs(y_new-ridge_pred_new)),4)
  out$inrmse <-round(sqrt(mean((y-ridge_pred)^2)),4)
  out$outrmse <-round(sqrt(mean((y_new-ridge_pred_new)^2)),4)
  return(out)
}


print("stage 3")

num_part<- 2000
num_test<- 2000
set.seed(JobId)
ind.to.use <- get_ind(num_part,num_test)

print("Standardisation transformation")

x.train.scaled <- dat_allmat[ind.to.use$train,]
x.test.scaled <-dat_allmat[ind.to.use$test,]

print("PCA")
pca.train <-prcomp(x.train.scaled,center = FALSE, scale = FALSE)
pca.train.sum <- summary(pca.train)

for(k in c(90)){

  num.sel.pc <- which(pca.train.sum$importance[3,]>k*0.01)[1]
  pca.train.sc <- pca.train$x[,1:num.sel.pc]
  
  pca.test.sc <- x.test.scaled %*% as.matrix(pca.train$rotation)
  pca.test.sc <- pca.test.sc[,1:num.sel.pc]
  
  print("=====")
  print("==========")
  print("fit model")
  
  lassofit <- lm(age_tab$age[ind.to.use$train] ~ as.matrix(pca.train.sc))
  print("in-predict")
  pred_prior<-predict(lassofit)
  print("out-predict")
  pred_prior_new<-predict(lassofit, as.data.frame(pca.test.sc))
  
  res_lm <- c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(coefficients(lassofit)[-1]))),sum(abs(coefficients(lassofit)[-1])>1e-5),num.sel.pc)
  
  write.csv(rbind(age_tab$age[ind.to.use$train],age_tab$age[ind.to.use$test],pred_prior,pred_prior_new), paste0("/well/nichols/users/qcv214/bnn2/smooth/pca_lm_pred_usc",k,"_",JobId,".csv"), row.names = FALSE)
  write.csv(c(res_lm), paste0("/well/nichols/users/qcv214/bnn2/smooth/pca_lm_usc",k,"_",JobId,".csv"), row.names = FALSE)
}

