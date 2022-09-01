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

res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age
#Length
n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(dat_allmat)
n.dat <- nrow(dat_allmat)

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
set.seed(4)
# ind.to.use <- get_ind(num_part,num_test)
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train <-  unlist(ind.temp[1,])

print("Standardisation transformation")
x.train.mean <- colMeans(dat_allmat[train.test.ind$train,],na.rm = TRUE)
x.train.sd <- apply(dat_allmat[train.test.ind$train,], 2, sd)
x.train.centred<-sweep(dat_allmat[train.test.ind$train,],2,x.train.mean,"-")
x.train.scaled <- sweep(x.train.centred,2,x.train.sd,"/")

x.test.centred <- sweep(dat_allmat[train.test.ind$test,],2,x.train.mean,"-")
x.test.scaled <- sweep(x.test.centred,2,x.train.sd,"/")

print("PCA")
pca.train <-prcomp(x.train.scaled,center = FALSE, scale = FALSE)
pca.train.sum <- summary(pca.train)

  num.sel.pc <- which(pca.train.sum$importance[3,]>0.90)[1]
  # num.sel.pc <- which(pca.train.sum$importance[3,]>0.50)[1] #CHANGED here at 1.43pm on 15 Feb to assess overfitting of 90%
  # num.sel.pc <- which(pca.train.sum$importance[3,]>0.25)[1]
  pca.train.sc <- pca.train$x[,1:num.sel.pc]
  
  pca.test.sc <- x.test.scaled %*% as.matrix(pca.train$rotation)
  pca.test.sc <- pca.test.sc[,1:num.sel.pc]
  
  print("=====")
  print("==========")
  print("fit model")
  
  lassofit <- lm(age_tab$age[train.test.ind$train] ~ as.matrix(pca.train.sc))
  print("in-predict")
  pred_prior<-predict(lassofit)
  print("out-predict")
  pred_prior_new<-t(as.matrix(cbind(1,pca.test.sc))%*%coefficients(lassofit))
  write.csv(rbind(age_tab$age[train.test.ind$train],age_tab$age[train.test.ind$test],pred_prior,pred_prior_new), paste0("/well/nichols/users/qcv214/bnn2/res3/pile/pca_lm_pred_",JobId,".csv"), row.names = FALSE)
  write.csv(c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = train.test.ind$train,ind.new = train.test.ind$test)))),as.numeric(sub('.*:', '', summary(coefficients(lassofit)[-1]))),sum(abs(coefficients(lassofit)[-1])>1e-5),num.sel.pc),
            paste0("/well/nichols/users/qcv214/bnn2/res3/pile/pca_lm_",JobId,".csv"), row.names = FALSE)
  
  print(dim(as.matrix(pca.train$rotation)[,1:num.sel.pc]))
  print(length(c(coefficients(lassofit)[-1])))
  rotated.beta <- as.matrix(pca.train$rotation)[,1:num.sel.pc] %*% c(coefficients(lassofit)[-1])
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <-abs(rotated.beta)
  gp.mask.hs@datatype = 16
  gp.mask.hs@bitpix = 32
  writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/proj_pca_',JobId))