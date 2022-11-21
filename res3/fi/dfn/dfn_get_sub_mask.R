#RScript

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

temp.img <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/res3mask.nii.gz')
temp.img[temp.img ==2] <- 0
temp.img[temp.img ==3] <- 0
temp.img[temp.img ==8] <- 0
temp.img[temp.img ==9] <- 0

temp.img@datatype = 2
temp.img@bitpix = 8
writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sub_res3mask'))


