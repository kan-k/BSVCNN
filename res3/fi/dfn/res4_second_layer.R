#For creating partial GP for centroids (hidden layer)

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

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
mask.reg <- sort(setdiff(unique(c(mask_subcor)),c(0)))

#load preset
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))

#Load centroid coords
coord.centroid <- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/res4_coord_centroids.csv")))
#Use coord.centroid[,5:7] for transformed coord

#Get partial GP
#param:
poly_degree = 10
a_concentration = 2
b_smoothness = 40
dimension = 3
#get psi
psi.mat.nb <- GP.eigen.funcs.fast(coord.centroid[,5:7], poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
#get lambda
lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
#Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
sqrt.lambda.nb <- sqrt(lambda.nb)
bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
partial.gp.centroid <- t(as.matrix(bases.nb))

