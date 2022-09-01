# R script
#devtools::install_github("kangjian2016/PMS", force = TRUE) 
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(dplyr)
##3 Dec with white matter, stem removed and thresholded


print("stage 1")
JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
part_1<-oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

#load age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age

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
  out$inmae<-round(mean(abs(y-ridge_pred)),4)
  out$outrsq<-round(1-sserr_ridge_new/sstot_new,5)*100
  out$outmae<-round(mean(abs(y_new-ridge_pred_new)),4)
  out$inrmse <-round(sqrt(mean((y-ridge_pred)^2)),4)
  out$outrmse <-round(sqrt(mean((y_new-ridge_pred_new)^2)),4)
  return(out)
}


print("stage 3")

num_part<- 2000
num_test<- 2000
set.seed(4)
ind.to.use <- get_ind(num_part,num_test)

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
ind.to.use <- list()
ind.to.use$test <- unlist(ind.temp[2,])
ind.to.use$train <- unlist(ind.temp[1,])

set.seed(NULL)
print("=====")
print("==========")
print("fitting hs")
time.train <-  Sys.time()
lassofit <- fast_horseshoe_lm(X = cbind(1,as.matrix(dat_allmat[ind.to.use$train,])) ,y = age_tab$age[ind.to.use$train],mcmc_sample = 400L) #Change smaples to 400
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

gp.mask.hs <- mask_subcor

gp.mask.hs[gp.mask.hs!=0] <-abs(lassofit$post_mean$betacoef[-1])

gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32

writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/hs_',JobId))
