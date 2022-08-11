if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(nimble)
p_load(resample)

print("stage 1")
JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

##################Load data
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

##################Load train-test
print("Load data")
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <- unlist(ind.temp[2,])
train.test.ind$train <- unlist(ind.temp[1,])


##################Calculate voxel-Mean
print("Calculate voxel-Mean")

vox.mean.train <- colMeans(dat_allmat[train.test.ind$train,])
vox.mean.test <- colMeans(dat_allmat[train.test.ind$test,])

gp.mask.hs <- mask.com
gp.mask.nm <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-c(vox.mean.train)
gp.mask.nm[gp.mask.nm!=0] <-c(vox.mean.test)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_mean_train'))
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_mean_test'))


##################Calculate voxel-variance
print("Calculate voxel-variance")
      
vox.var.train <- colVars(dat_allmat[train.test.ind$train,])
vox.var.test <- colVars(dat_allmat[train.test.ind$test,])

gp.mask.hs <- mask.com
gp.mask.nm <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-c(vox.var.train)
gp.mask.nm[gp.mask.nm!=0] <-c(vox.var.test)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_var_train'))
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_var_test'))

##################Calculate label-variance of voxel-mean
print("Calculate label-variance of voxel-mean")

####Train
subcor_masked<-mask.com[mask.com>0]
###Put labels (sub coritcal) to masked vbm_means
vmb_means_masked_lab<-cbind(subcor_masked,vox.mean.train)
vmb_means_masked_lab<-as.data.frame(vmb_means_masked_lab)
vmb_means_masked_lab$subcor_masked<-as.factor(vmb_means_masked_lab$subcor_masked)
###Find label-level variance
lab_var_ref<-aggregate(vmb_means_masked_lab[,2], list(vmb_means_masked_lab$subcor_masked), var)
#label variance
lab_var.mean.train<-matrix(,nrow=1,ncol=length(vox.mean.train))
for(i in 1:length(vox.mean.train)){
  lab_var.mean.train[1,i]<-lab_var_ref[lab_var_ref$Group.1==subcor_masked[i],2]
}

####Test
subcor_masked<-mask.com[mask.com>0]
###Put labels (sub coritcal) to masked vbm_means
vmb_means_masked_lab<-cbind(subcor_masked,vox.mean.test)
vmb_means_masked_lab<-as.data.frame(vmb_means_masked_lab)
vmb_means_masked_lab$subcor_masked<-as.factor(vmb_means_masked_lab$subcor_masked)
###Find label-level variance
lab_var_ref<-aggregate(vmb_means_masked_lab[,2], list(vmb_means_masked_lab$subcor_masked), var)
#label variance
lab_var.mean.test<-matrix(,nrow=1,ncol=length(vox.mean.test))
for(i in 1:length(vox.mean.test)){
  lab_var.mean.test[1,i]<-lab_var_ref[lab_var_ref$Group.1==subcor_masked[i],2]
}

gp.mask.hs <- mask.com
gp.mask.nm <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-c(lab_var.mean.train)
gp.mask.nm[gp.mask.nm!=0] <-c(lab_var.mean.test)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_reg_var_mean_train'))
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_reg_var_mean_test'))

##################Calculate label-mean of voxel-variance
print("Calculate label-mean of voxel-variance")

####Train
subcor_masked<-mask.com[mask.com>0]
###Put labels (sub coritcal) to masked vbm_means
vmb_means_masked_lab<-cbind(subcor_masked,vox.var.train)
vmb_means_masked_lab<-as.data.frame(vmb_means_masked_lab)
vmb_means_masked_lab$subcor_masked<-as.factor(vmb_means_masked_lab$subcor_masked)
###Find label-level variance
lab_var_ref<-aggregate(vmb_means_masked_lab[,2], list(vmb_means_masked_lab$subcor_masked), mean)
#label variance
lab_mean.var.train<-matrix(,nrow=1,ncol=length(vox.var.train))
for(i in 1:length(vox.var.train)){
  lab_mean.var.train[1,i]<-lab_var_ref[lab_var_ref$Group.1==subcor_masked[i],2]
}

####Test
subcor_masked<-mask.com[mask.com>0]
###Put labels (sub coritcal) to masked vbm_means
vmb_means_masked_lab<-cbind(subcor_masked,vox.var.test)
vmb_means_masked_lab<-as.data.frame(vmb_means_masked_lab)
vmb_means_masked_lab$subcor_masked<-as.factor(vmb_means_masked_lab$subcor_masked)
###Find label-level variance
lab_var_ref<-aggregate(vmb_means_masked_lab[,2], list(vmb_means_masked_lab$subcor_masked), mean)
#label variance
lab_mean.var.test<-matrix(,nrow=1,ncol=length(vox.var.test))
for(i in 1:length(vox.var.test)){
  lab_mean.var.test[1,i]<-lab_var_ref[lab_var_ref$Group.1==subcor_masked[i],2]
}

gp.mask.hs <- mask.com
gp.mask.nm <- mask.com
gp.mask.hs[gp.mask.hs!=0] <-c(lab_mean.var.train)
gp.mask.nm[gp.mask.nm!=0] <-c(lab_mean.var.test)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
gp.mask.nm@datatype = 16
gp.mask.nm@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_reg_mean_var_train'))
writeNIfTI(gp.mask.nm,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_reg_mean_var_test'))

