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
p_load(extraDistr)
p_load(dplyr)
p_load(moments)
p_load(stats)
JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)
set.seed(JobId)


res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
colnames(res3.dat) <- NULL

ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <-  unlist(ind.temp[2,])
train.test.ind$train <-  unlist(ind.temp[1,])
n.train <- length(train.test.ind$train)

res.dat.train <- res3.dat[train.test.ind$train,]
shap <- unlist(apply(res.dat.train,2,shapiro.test))
print(length(shap))
print(head(shap))

gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- as.numeric(shap[names(shap)=="p.value"])
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_sw_pvalue_train'))
gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- as.numeric(shap[names(shap)=="statistic.W"])
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_sw_teststat_train'))

res.dat.test <- res3.dat[train.test.ind$test,]
shap <- unlist(apply(res.dat.test,2,shapiro.test))
gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- as.numeric(shap[names(shap)=="p.value"])
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_sw_pvalue_test'))
gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- as.numeric(shap[names(shap)=="statistic.W"])
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/res3_vox_sw_teststat_test'))




