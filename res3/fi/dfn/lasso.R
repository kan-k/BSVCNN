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
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))
mask.com<- oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sub_res3mask.nii.gz')
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use,'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz')
# read multiple image files on brain mask
dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sub_res3mask'))
#load age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/fi.feather')
dat.age <- age_tab$fi
print("stage 2")

#func

rsqcal2<-function(old,new,ind.old,ind.new){
  ridge_pred<-old
  ridge_pred_new<-new
  no<-length(old)
  nn<-length(new)
  #insample
  y<-age_tab$fi[ind.old]
  sserr_ridge<-sum((y-ridge_pred)^2)
  sstot<-var(y)*length(y)
  #outsample
  y_new<-age_tab$fi[ind.new]
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
ind.to.use<- list()
ind.to.use$test <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index.csv')$x
ind.to.use$train <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index.csv')$x
#Tested.... doing set seed(4) get_ind is the same as loading sim_wb2_index


#get beta(v)
time.train <-  Sys.time()
lassofit<- cv.glmnet(x = as.matrix(dat_allmat[ind.to.use$train, ]) ,y = dat.age[ind.to.use$train], alpha = 1, lambda = NULL) #alpha does matter here, 0 is ridge
pred_prior<-predict(lassofit, as.matrix(dat_allmat[ind.to.use$train,]), s= "lambda.min")
pred_prior_new<-predict(lassofit, as.matrix(dat_allmat[ind.to.use$test, ]), s= "lambda.min")

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv( c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(coef(lassofit, s=lassofit$lambda.min)[-1,]))),sum(abs(coef(lassofit, s=lassofit$lambda.min))>1e-8)),
          paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_nov11_lasso_noscale_",JobId,".csv"), row.names = FALSE)

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_nov11_lasso_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_nov11_lasso_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use

gp.mask.hs <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-abs(coef(lassofit, s=lassofit$lambda.min)[-1,])

gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32


writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/viz/lasso_',JobId))