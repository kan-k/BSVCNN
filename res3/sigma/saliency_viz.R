if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

#Choose seed 21
##########Mean map of saliency
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
p.dat <- sum(res3.mask>0)
for(e in c(80)){
  saliency.array <- as.matrix(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/re_oct7_stgp2_saliencymatrix_seed_',15,"_epoch_",e)))
}
saliency.array.m <-apply(saliency.array, c(2), mean)
#Box plot

#Save viz
for(e in c(80)){
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <- abs(saliency.array.m)
  gp.mask.hs@datatype = 16
  gp.mask.hs@bitpix = 32
  writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/oct10_stgp2_saliency_abs_mean_seed_',15,"_epoch_",e))
}

####### SD map of saliency
# for(e in c(80)){
#   saliency.array <- as.matrix(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/re_oct7_stgp2_saliencymatrix_seed_',15,"_epoch_",e)))
# }
saliency.array.m <-apply(saliency.array, c(2), sd)
# saliency.array.m <- log(saliency.array.m + 1e-13)

#Box plot
# abline(h=0.0045,col='red')
#Save viz
for(e in c(80)){
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <- saliency.array.m
  gp.mask.hs@datatype = 16
  gp.mask.hs@bitpix = 32
  writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/oct10_stgp2_saliency_sd_seed_',15,"_epoch_",e))
}

#voxel 43 56 41