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
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)


mask.com <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(mask.com)),0))
#data
###############
#data
dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
n.dat.ori <-nrow(dat_allmat)
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age_tab <- age_tab[order(age_tab$id),]

num.add <- 4000
part_use<- (read.csv('/well/nichols/users/qcv214/bnn2/add_1_part_id_use_final.txt')$V1)[1:num.add]
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
dat_allmat <- rbind(dat_allmat,as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')))
age_tab<- rbind(age_tab,(read_feather('/well/nichols/users/qcv214/bnn2/res3/age_add1.feather'))[1:num.add,])

dat.age <-age_tab$age

p.dat <- ncol(dat_allmat)
n.dat <- nrow(dat_allmat)

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train.original <-  unlist(ind.temp[1,])
train.test.ind$train <- c(train.test.ind$train.original,(n.dat.ori+1):(n.dat.ori+num.add))


###############
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

#Tested.... doing set seed(4) get_ind is the same as loading sim_wb2_index


#get beta(v)
time.train <-  Sys.time()
print("fitting")
print(dim(data.matrix(as.matrix(dat_allmat))))
lassofit<- cv.glmnet(x = data.matrix(as.matrix(dat_allmat[train.test.ind$train, ])) ,y = dat.age[train.test.ind$train], alpha = 0, lambda = NULL,standardize = FALSE) #alpha does matter here, 0 is ridge
print("in-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[train.test.ind$train,]))))
pred_prior<-predict(lassofit, data.matrix(as.matrix(dat_allmat[train.test.ind$train,])), s= "lambda.min")
print("out-predicting")
print(dim(data.matrix(as.matrix(dat_allmat[train.test.ind$test,]))))
pred_prior_new<-predict(lassofit, data.matrix(as.matrix(dat_allmat[train.test.ind$test,])), s= "lambda.min")

time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)

write.csv( c(unlist(t(as.matrix(rsqcal2(pred_prior,pred_prior_new,ind.old = train.test.ind$train,ind.new = train.test.ind$test)))),as.numeric(sub('.*:', '', summary(coef(lassofit, s=lassofit$lambda.min)[-1,]))),sum(abs(coef(lassofit, s=lassofit$lambda.min))>1e-8)),
           paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_may1_ridge_add4k_noscale_",JobId,".csv"), row.names = FALSE)

print("write prediction")

write.csv(c(pred_prior_new),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_may1_ridge_add4k_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/re_may1_ridge_add4k_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use

gp.mask.hs <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-abs(coef(lassofit, s=lassofit$lambda.min)[-1,])

gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32


writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/may1_ridge_add4k_',JobId))