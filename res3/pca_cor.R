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
res3.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather'))
#Age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
age <- age_tab$age
#Length
n.mask <- length(res3.mask.reg)
n.expan <- choose(10+3,3)
p.dat <- ncol(res3.dat)
n.dat <- nrow(res3.dat)

print("Standardisation transformation")
x.train.mean <- colMeans(res3.dat,na.rm = TRUE)
x.train.sd <- apply(res3.dat, 2, sd)
x.train.centred<-sweep(res3.dat,2,x.train.mean,"-")
x.train.scaled <- sweep(x.train.centred,2,x.train.sd,"/")

print("PCA")
pca.train <-prcomp(x.train.scaled,center = FALSE, scale = FALSE)
pca.train.sum <- summary(pca.train)
num.sel.pc <- which(pca.train.sum$importance[3,]>0.90)[1]
pca.train.sc <- pca.train$x[,1:num.sel.pc]

print("Computing pair-wise correlation")
pair.wise.cor <- cor(res3.dat,pca.train.sc)
max.cor <- apply(pair.wise.cor, 1, max)
max.cor.abs <- apply(abs(pair.wise.cor),1,max)

gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- c(max.cor)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
print("finish writing 1")
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/pca_correlation'))

gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- c(max.cor.abs)
gp.mask.hs@datatype = 16
gp.mask.hs@bitpix = 32
print("finish writing 2")
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/viz/pca_correlation_abs'))



