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

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bsvcnn/pile/HO-combined-fin.nii.gz')
mask.reg <- sort(setdiff(unique(c(mask_subcor)),c(0)))
mask.reg.part<-split(mask.reg, sort(mask.reg%%100))

#load preset
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
part_1<-oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

poly_degree = 10
a_concentration= 0.01
b_smoothness= 0.001
# poly_degree = 40
# a_concentration= 0.001
# b_smoothness= 0.1

for(k in mask.reg.part[[JobId]]){
  
  mask_subcor<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bsvcnn/pile/combined/fin_mask_ROI_',k))
  nb <- find_brain_image_neighbors(img1, mask_subcor, radius=1)
  
  #re-centre each region
  # nb.centred <- sweep(nb$maskcoords,2,apply(nb$maskcoords,2,median),"-") #Change on 18
  #re-scale each region
  nb.centred <- apply(nb$maskcoords,2,norm.func)

  
  #get psi
  psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = 3)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
  
  set.seed(2022)
  theta <- rnorm(ncol(psi.mat.nb))
  f <- crossprod(bases.nb,theta)
  
  #plot f back to brain
  gp.mask <- mask_subcor
  gp.mask[gp.mask>0] <- f
  gp.mask@datatype = 16
  gp.mask@bitpix = 32
  writeNIfTI(gp.mask,paste0('/well/nichols/users/qcv214/bnn2/pile/mask_ROI_',k,'_gp_',poly_degree,a_concentration,b_smoothness))
  print(k)
}
