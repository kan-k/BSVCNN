if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

start.time <- Sys.time()
print("Loading data")

#Load data and mask and GP 
#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age
#Length
n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

time.taken <- Sys.time() - start.time
cat("Loading mask, age and vbm complete in: ", time.taken)

start.time <- Sys.time()
print("Loading first layer GP")
source("/well/nichols/users/qcv214/bnn2/res3/first_layer_gp.R")
print(dim(partial.gp))
time.taken <- Sys.time() - start.time
cat("Loading 1st-layer GP complete in: ", time.taken)

start.time <- Sys.time()
print("Loading 2nd-layer GP")
partial.gp.centroid<-t(as.matrix(read_feather(paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_centroids_fixed_100.540.feather"))))
time.taken <- Sys.time() - start.time
cat("Loading 2nd-layer GP complete in: ", time.taken)

sum(is.na(c(partial.gp[1,1:100,1:100])))
