# R script
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

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
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
dat.age <-age_tab$age

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
norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

print("stage 3")


poly_degree = 10
a_concentration = 0.5
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
num_part<- 2000
num_test<- 2000
set.seed(4)
ind.to.use <- get_ind(num_part,num_test)

train_Y <- dat.age[ind.to.use$train]
train_Z <- z.nb[ind.to.use$train,]
test_Y <- dat.age[ind.to.use$test]
test_img <- dat_allmat[ind.to.use$test,]

#get beta(v)
hs_fit_SOI <- fast_horseshoe_lm(train_Y,train_Z)
nm_fit_SOI <- fast_normal_lm(train_Y,train_Z)

beta_fit <- data.frame(HS = crossprod(bases.nb,hs_fit_SOI$post_mean$betacoef[-1]),
                       NM = crossprod(bases.nb,nm_fit_SOI$post_mean$betacoef[-1]))

# mask.reg <- sort(setdiff(unique(c(mask.com)),c(0)))
# reg.ref <- vector(mode = "integer")
# for(i in sort(mask.reg)){
#   mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bsvcnn/pile/combined/fin_mask_ROI_',i))
#   reg.ref <- c(reg.ref,rep(i, sum(mask.temp!=0)))
# }

gp.mask.hs <- mask.com
gp.mask.nm <- mask.com

gp.mask.hs[gp.mask.hs!=0] <-beta_fit$HS
gp.mask.nm[gp.mask.nm!=0] <-beta_fit$NM
# for(i in sort(mask.reg)){
#   print(i)
#   gp.mask.hs[gp.mask.hs==i] <- beta_fit$HS[reg.ref==i]
#   gp.mask.nm[gp.mask.nm==i] <- beta_fit$NM[reg.ref==i]
#   
# }
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32

writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/wb_gp_hs_',poly_degree,a_concentration,b_smoothness))
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/wb_gp_nm_',poly_degree,a_concentration,b_smoothness))

