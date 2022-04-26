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
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

#Load centroid coords
coord.centroid <- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/res4_coord_centroids.csv")))
#Use coord.centroid[,5:7] for transformed coord

#Get partial GP
#param:
poly_degree = 10
a_concentration = 0.5
b_smoothness = 40
dimension = 3
#get psi
psi.mat.nb <- GP.eigen.funcs.fast(coord.centroid[,5:7], poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
#get lambda
lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
#Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
sqrt.lambda.nb <- sqrt(lambda.nb)
bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
bases.nb <- as.matrix(bases.nb)

write_feather(as.data.frame(bases.nb),paste0("/well/nichols/users/qcv214/bnn2/res3/roi/res4_partial_gp_centroids_fixed_",poly_degree,a_concentration,b_smoothness,".feather"))
