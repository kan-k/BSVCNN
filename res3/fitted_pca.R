# R script
#Change for loop for each batch update to matrix (Hadamard) update
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))
print("Starting")

#Load data and mask and GP 
#mask
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(res3.mask)),0))
#data
dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age
#Length
n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(dat_allmat)
n.dat <- nrow(dat_allmat)

get_ind <- function(num.train,num.test = 1){
  full.ind<-1:dim(dat_allmat)[1]
  train<-sample(x = full.ind,size = num.train,replace = FALSE)
  test<-sample(x=setdiff(full.ind,train),size=num.test,replace=FALSE)
  out=list()
  out$train<-train
  out$test<-test
  return(out)
}

num_part<- 2000
num_test<- 2000
set.seed(4)
ind.to.use <- get_ind(num_part,num_test)

print("Standardisation transformation")
x.train.mean <- colMeans(dat_allmat[ind.to.use$train,],na.rm = TRUE)
x.train.sd <- apply(dat_allmat[ind.to.use$train,], 2, sd)
x.train.centred<-sweep(dat_allmat[ind.to.use$train,],2,x.train.mean,"-")
x.train.scaled <- sweep(x.train.centred,2,x.train.sd,"/")

print("PCA")
pca.train <-prcomp(x.train.scaled,center = FALSE, scale = FALSE)
pca.train.sum <- summary(pca.train)
num.sel.pc <- which(pca.train.sum$importance[3,]>0.90)[1]
pca.train.sc <- pca.train$x[,1:num.sel.pc]

lassofit <- lm(age_tab$age[ind.to.use$train] ~ as.matrix(pca.train.sc))

gp.mask.hs <- res3.mask
print(dim(as.matrix(pca.train$rotation)))
print(length(coef(lassofit)[-1]))
gp.mask.hs[gp.mask.hs!=0] <- c(as.matrix(pca.train$rotation[,1:num.sel.pc]) %*% coef(lassofit)[-1])
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
print("finish writing 1")
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/fitted_pca'))