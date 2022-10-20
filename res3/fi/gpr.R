# R script

#I have changed the way GPR is fitted 

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

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
part_1<-oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

mask.com<- oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# read multiple image files on brain mask
dat_allmat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask'))

nb <- find_brain_image_neighbors(img1, mask.com, radius=1)


#load age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/fi/fi.feather')
dat.age <- age_tab$fi
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
norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

print("stage 3")
time.train <-  Sys.time()
poly_degree = 10
a_concentration = 2
b_smoothness = 40

nb.centred<- apply(nb$maskcoords,2,norm.func)
#get psi
psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
#get lambda
lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = 3)
#Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
sqrt.lambda.nb <- sqrt(lambda.nb)
bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb

print("before cbind")
#Get design matrix
z.nb <- cbind(1,t(bases.nb%*%t(dat_allmat)))
print("after cbind")

#subset data
ind.to.use<- list()

ind.to.use$train <- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/train_index.csv')$x
ind.to.use$test <- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/test_index.csv')$x
#Tested.... doing set seed(4) get_ind is the same as loading sim_wb2_index

train_Y <- dat.age[ind.to.use$train]
train_Z <- z.nb[ind.to.use$train,]
test_Y <- dat.age[ind.to.use$test]
test_img <- dat_allmat[ind.to.use$test,]
test_Z <- z.nb[ind.to.use$test,]

#get beta(v)
time.train <-  Sys.time()
lassofit <- fast_horseshoe_lm(X = train_Z ,y =train_Y,mcmc_sample = 1000L) #Change smaples to 400
time.taken <- Sys.time() - time.train
cat("Training complete in: ", time.taken)
pred_prior<-predict_fast_lm(lassofit, train_Z )#$mean
pred_prior_new<-predict_fast_lm(lassofit, test_Z)#$mean

write.csv(c(unlist(t(as.matrix(rsqcal2(pred_prior$mean,pred_prior_new$mean,ind.old = ind.to.use$train,ind.new = ind.to.use$test)))),as.numeric(sub('.*:', '', summary(lassofit$post_mean$betacoef[-1,]))),sum(abs(lassofit$post_mean$betacoef[-1,])>1e-5)),
          paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior_new$mean),paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_outpred_noscale_",JobId,".csv"), row.names = FALSE)
write.csv(c(pred_prior$mean),paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_inpred_noscale_",JobId,".csv"), row.names = FALSE)
####Result to use
write.csv(rbind(c(ind.to.use$train),c(ind.to.use$test)),paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_index_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(lassofit$post_mean$betacoef),paste0( '/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_coef_',JobId,'.feather'))




#Posterior Predictive mean in-sampleMSE
in.mse <- mean((pred_prior$mean-age_tab$fi[ind.to.use$train])^2)

#######In-sample################################################
stat.in.ig.stderrmod <- sqrt(in.mse+pred_prior$sd^2)
stat.in.ig.lwrmod <- pred_prior$mean - qt(0.975,1000-1)*stat.in.ig.stderrmod
stat.in.ig.uprmod <- pred_prior$mean + qt(0.975,1000-1)*stat.in.ig.stderrmod
##for no personalised variance
stat.in.ig.lwrmod2 <- pred_prior$mean - qt(0.975,1000-1)*sqrt(in.mse)
stat.in.ig.uprmod2 <- pred_prior$mean + qt(0.975,1000-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$fi[ind.to.use$train])){
  within.pred <- c(within.pred,(between(age_tab$fi[ind.to.use$train][i],stat.in.ig.lwrmod[i],stat.in.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$fi[ind.to.use$train][i],stat.in.ig.lwrmod2[i],stat.in.ig.uprmod2[i])))
}
stat.in.ig.true.covermod <- within.pred
stat.in.ig.true.covermod2 <- within.pred2

print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.in.ig.true.covermod)/1839*100))

#######Out-of-sample################################################
stat.out.ig.stderrmod <- sqrt(in.mse+pred_prior_new$sd^2)
stat.out.ig.lwrmod <- pred_prior_new$mean - qt(0.975,1000-1)*stat.out.ig.stderrmod
stat.out.ig.uprmod <- pred_prior_new$mean + qt(0.975,1000-1)*stat.out.ig.stderrmod
##for no personalised variance
stat.out.ig.lwrmod2 <- pred_prior_new$mean - qt(0.975,1000-1)*sqrt(in.mse)
stat.out.ig.uprmod2 <- pred_prior_new$mean + qt(0.975,1000-1)*sqrt(in.mse)

#Define proportion counting

#True
within.pred <- vector(mode='numeric')
within.pred2 <- vector(mode='numeric')
for(i in 1:length(age_tab$fi[ind.to.use$test])){
  within.pred <- c(within.pred,(between(age_tab$fi[ind.to.use$test][i],stat.out.ig.lwrmod[i],stat.out.ig.uprmod[i])))
  within.pred2 <- c(within.pred2,(between(age_tab$fi[ind.to.use$test][i],stat.out.ig.lwrmod2[i],stat.out.ig.uprmod2[i])))
  
}
stat.out.ig.true.covermod <- within.pred
stat.out.ig.true.covermod2 <- within.pred2
print(paste0('Proprtion of true lying within subject 95% prediction interval: ',sum(stat.out.ig.true.covermod)/1869*100))

cover.mat <- matrix(c(sum(stat.in.ig.true.covermod)/1839,sum(stat.out.ig.true.covermod)/1869,sum(stat.in.ig.true.covermod2)/1839,sum(stat.out.ig.true.covermod2)/1869),ncol = 4)*100
colnames(cover.mat) <- c("train","test","npvtrain","npvtest")

write.csv(cover.mat,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_oct12_gpr_coverage_",JobId,".csv"), row.names = FALSE)