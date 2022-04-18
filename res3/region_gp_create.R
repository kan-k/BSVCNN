# R script to compute GP first layer and 2nd layer
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)


#First layer, compute 12 gps, mask already have loaded mask/mask.reg AND have mask_ROI ready to load
dimension = 3
deg_vec<- c(10)
a_vec  <- c(0.001,0.01,0.1,0.5,1,2,3)
b_vec  <- 40
param_grid <- as.matrix(expand.grid(deg_vec,a_vec,b_vec))

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
masked.mask <- mask_subcor[mask_subcor > 0]
res3.mask.reg <- sort(setdiff(unique(c(mask_subcor)),0))

#load preset
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))


gs.opt <- matrix(,ncol=2,nrow = length(res3.mask.reg))
for(i in sort(res3.mask.reg)){
  gs.opt[which(i==(res3.mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/gs5april_ROI_",i,".csv")))
}

# bases.nb.temp <- matrix(,ncol=1,nrow= choose(max(deg_vec)+dimension,dimension))
norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

partial.gp <- array(, dim = c(length(res3.mask.reg),p.dat,n.expan))
for(i in res3.mask.reg){
  poly_degree = 10
  a_concentration = param_grid[gs.opt[which(i==(res3.mask.reg)),1],2]
  b_smoothness = 40
  
  #Load region mask
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/roi/mask_ROI_',i))
  nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)
  #re-scale the coordinates
  nb.centred<- apply(nb$maskcoords,2,norm.func)
  #get psi
  psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(as.matrix(t(psi.mat.nb)*sqrt.lambda.nb))
  
  full.bases <- matrix(0,nrow=p.dat,ncol=n.expan)
  # print(dim(bases.nb))
  # print(sum(c(mask_subcor)==i))
  mask.ind <- which(c(masked.mask)==i)
  # print(dim(full.bases))
  # print(length(mask.ind))
  # print(max(mask.ind))
  # print(min(mask.ind))
  full.bases[mask.ind,] <- bases.nb
  partial.gp[i,,] <- full.bases
}