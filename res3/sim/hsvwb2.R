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

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

dat_allmat<-as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/sim/sub_res3_dat.feather'))

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
part_1<-oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res3_sub.nii.gz')
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

#load age
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
pred_prior<-predict_fast_lm(lassofit, cbind(1,as.matrix(dat_allmat[ind.to.use$train, ])))#$mean
pred_prior_new<-predict_fast_lm(lassofit, cbind(1,as.matrix(dat_allmat[ind.to.use$test, ])))#$mean

write.csv(c(unlist(t(as.matrix(rsqcal2(pred_prior$mean,pred_prior_new$mean,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(lassofit$post_mean$betacoef[-1,]))),sum(abs(lassofit$post_mean$betacoef[-1,])>1e-5)),
          paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior_new$mean),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior$mean),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use
write.csv(rbind(c(ind.to.use$train),c(ind.to.use$test)),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_index_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(lassofit$post_mean$betacoef),paste0( '/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_coef_',JobId,'.feather'))

#Posterior Predictive Mean
# pred_prior$mean
# pred_prior_new$mean

#MCMC s.d.
# pred_prior$sd
# pred_prior_new$sd

#Posterior Predictive mean in-sampleMSE
in.mse <- mean((pred_prior$mean-age_tab$age[ind.to.use$train])^2)

#######In-sample################################################
stat.in.ig.stderrmod <- sqrt(in.mse+pred_prior$sd^2)
stat.in.ig.lwrmod <- pred_prior$mean - qt(0.975,500-1)*stat.in.ig.stderrmod
stat.in.ig.uprmod <- pred_prior$mean + qt(0.975,500-1)*stat.in.ig.stderrmod
##for no personalised variance
stat.in.ig.lwrmod2 <- pred_prior$mean - qt(0.975,500-1)*sqrt(in.mse)
stat.in.ig.uprmod2 <- pred_prior$mean + qt(0.975,500-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$age[ind.to.use$train])){
  within.pred <- c(within.pred,(between(age_tab$age[ind.to.use$train][i],stat.in.ig.lwrmod[i],stat.in.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$age[ind.to.use$train][i],stat.in.ig.lwrmod2[i],stat.in.ig.uprmod2[i])))
}
stat.in.ig.true.covermod <- within.pred
stat.in.ig.true.covermod2 <- within.pred2

print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.in.ig.true.covermod)/2000*100))

#######Out-of-sample################################################
stat.out.ig.stderrmod <- sqrt(in.mse+pred_prior_new$sd^2)
stat.out.ig.lwrmod <- pred_prior_new$mean - qt(0.975,500-1)*stat.out.ig.stderrmod
stat.out.ig.uprmod <- pred_prior_new$mean + qt(0.975,500-1)*stat.out.ig.stderrmod
##for no personalised variance
stat.out.ig.lwrmod2 <- pred_prior_new$mean - qt(0.975,500-1)*sqrt(in.mse)
stat.out.ig.uprmod2 <- pred_prior_new$mean + qt(0.975,500-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$age[ind.to.use$test])){
  within.pred <- c(within.pred,(between(age_tab$age[ind.to.use$test][i],stat.out.ig.lwrmod[i],stat.out.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$age[ind.to.use$test][i],stat.out.ig.lwrmod2[i],stat.out.ig.uprmod2[i])))
  
}
stat.out.ig.true.covermod <- within.pred
stat.out.ig.true.covermod2 <- within.pred2
print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.out.ig.true.covermod)/2000*100))

cover.mat <- matrix(c(sum(stat.in.ig.true.covermod),sum(stat.out.ig.true.covermod),sum(stat.in.ig.true.covermod2),sum(stat.out.ig.true.covermod2)),ncol = 4)/2000*100
colnames(cover.mat) <- c("train","test","npvtrain","npvtest")

write.csv(cover.mat,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_aug10_hsvwb_coverage_",JobId,".csv"), row.names = FALSE)
