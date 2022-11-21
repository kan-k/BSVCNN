# R script

#lasso 
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(dplyr)

##3 Dec with white matter, stem removed and thresholded

print("stage 1")
JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))



######Update 15 Nov, i think one participant from held-out dropped out, 4262 instead of 4263
# part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
# part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# #These two are equal
# part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# # 
# part_1<-oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# 
# mask.com<- oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# # read multiple image files on brain mask
# dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask'))
# #load age
# age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
# dat.age <-age_tab$age

################################################
mask.com <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(mask.com)),0))
#data
dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age_tab <- age_tab[order(age_tab$id),] ####NOTE HERE
dat.age <-age_tab$age

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train <-  unlist(ind.temp[1,])

print("stage 2")

#func

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
time.train <-  Sys.time()


#subset data
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
ind.to.use <- list()
ind.to.use$test <- unlist(ind.temp[2,])
ind.to.use$train <- unlist(ind.temp[1,])
#Tested.... doing set seed(4) get_ind is the same as loading sim_wb2_index


#get beta(v)
time.train <-  Sys.time()
print("fitting")
print(dim(data.matrix(as.matrix(dat_allmat))))
lassofit<- cv.glmnet(x = data.matrix(as.matrix(dat_allmat[ind.to.use$train, ])) ,y = dat.age[ind.to.use$train], alpha = 1, lambda = NULL) #alpha does matter here, 0 is ridge
print("in-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$train,]))))
pred_prior<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$train,])), s= "lambda.min")
print("out-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[ind.to.use$test,]))))
pred_prior_new<-predict(lassofit, data.matrix(as.matrix(dat_allmat[ind.to.use$test,])), s= "lambda.min")



time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv( c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(coef(lassofit, s=lassofit$lambda.min)[-1,]))),sum(abs(coef(lassofit, s=lassofit$lambda.min))>1e-8)),
           paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_nov14_lasso_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_nov14_lasso_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_nov14_lasso_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use

gp.mask.hs <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-abs(coef(lassofit, s=lassofit$lambda.min)[-1,])

gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32


writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/lasso_',JobId))