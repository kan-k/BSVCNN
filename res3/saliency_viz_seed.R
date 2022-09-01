if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

#Use all seed

##########Mean map of saliency
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
p.dat <- sum(res3.mask>0)

seed <- c(12,13,21,22,38,44,45)
for(s in seed){

saliency.array <- array(,dim=c(3,2000,p.dat))
for(e in c(25,50,70)){
  saliency.array[which(e==c(25,50,70)),,] <- as.matrix(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/re_aug27_saliencymatrix_seed_',s,"_epoch_",e)))
}
saliency.array.m <-apply(saliency.array, c(1,3), mean)
# saliency.array.m <- log(saliency.array.m + 1e-13)
for(e in c(25,50,70)){
  print(summary(saliency.array.m[which(e==c(25,50,70)),]))
}
#Box plot
boxplot(t(saliency.array.m), ylab="sd",main = "Voxel saliency mean by number of epochs", xlab="epochs")
#Save viz
for(e in c(25,50,70)){
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <- saliency.array.m[which(e==c(25,50,70)),]
  gp.mask.hs@datatype = 16
  gp.mask.hs@bitpix = 32
  writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_mean_seed_',s,"_epoch_",e))
}

####### SD map of saliency
saliency.array <- array(,dim=c(3,2000,p.dat))
for(e in c(25,50,70)){
  saliency.array[which(e==c(25,50,70)),,] <- as.matrix(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/re_aug27_saliencymatrix_seed_',s,"_epoch_",e)))
}
saliency.array.m <-apply(saliency.array, c(1,3), sd)
# saliency.array.m <- log(saliency.array.m + 1e-13)
for(e in c(25,50,70)){
  print(summary(saliency.array.m[which(e==c(25,50,70)),]))
}
#Box plot
boxplot(t(saliency.array.m), ylab="sd",main = "Voxel saliency sd by number of epochs", xlab="epochs")
abline(h=0.0005,col='red')
# abline(h=0.0045,col='red')
#Save viz
for(e in c(25,50,70)){
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <- saliency.array.m[which(e==c(25,50,70)),]
  gp.mask.hs@datatype = 16
  gp.mask.hs@bitpix = 32
  writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_sd_seed_',s,"_epoch_",e))
}

}
