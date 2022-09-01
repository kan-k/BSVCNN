#Absolute (of mean) Saliency of Seed 44

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)


seed<-45
for(e in c(25,50,70)){
  temp.sal <- abs(oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_mean_seed_',seed,"_epoch_",e,".nii.gz")))
  # writeNIfTI(temp.sal,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_abs_mean_seed_',seed,"_epoch_",e))
}


seed<-45
sal.thresh <- 0.005
for(e in c(25,50,70)){
  temp.sal <- abs(oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_mean_seed_',seed,"_epoch_",e,".nii.gz")))
  writeNIfTI(temp.sal,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/saliency_abs_mean_seed_',seed,"_epoch_",e))
  # print(sum(c(temp.sal)>0))
  sal.masked <- c(temp.sal)[(c(temp.sal)>0)]
  print(round(sum(sal.masked>0.005)/length(sal.masked)*100,3))
}