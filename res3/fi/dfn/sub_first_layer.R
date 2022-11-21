# R script to compute GP first layer and 2nd layer

#2nd layer with 2,3,8,9 removed


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
deg_vec<- c(6)
a_vec  <- c(1,2,3,4,5)
b_vec  <- c(30,40,50,60,70,80,90,100)
param_grid <- as.matrix(expand.grid(deg_vec,a_vec,b_vec))

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sub_res3mask.nii.gz')
res3.mask.reg <- sort(setdiff(unique(c(mask_subcor)),0))
masked.reg <- mask_subcor[mask_subcor>0]
#load preset
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))

gs.opt <- matrix(,ncol=2,nrow = length(res3.mask.reg))
for(i in sort(res3.mask.reg)){
  gs.opt[which(i==(res3.mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/dfn_gs19oct_ROI_",i,".csv")))
}

# bases.nb.temp <- matrix(,ncol=1,nrow= choose(max(deg_vec)+dimension,dimension))



partial.gp <- array(, dim = c(length(res3.mask.reg),p.dat,n.expan))
for(i in res3.mask.reg){
  #Load region mask
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/roi/dfn_mask_ROI_',i))
  invisible(capture.output(nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)))
  #Get the centre of the region
  nb.centre<- apply(nb$maskcoords,2,median)
  #Load full mask
  
  #get coord wrt full mask
  invisible(capture.output(nb.full <- find_brain_image_neighbors(img1, mask_subcor, radius=1)))
  #Re-centre the coords wrt centre of ROI
  nb.full$maskcoords[,1] <- nb.full$maskcoords[,1] - nb.centre[1]
  nb.full$maskcoords[,2] <- nb.full$maskcoords[,2] - nb.centre[2]
  nb.full$maskcoords[,3] <- nb.full$maskcoords[,3] - nb.centre[3]
  #Get max and min coord
  nb.xmax <- max(nb.full$maskcoords[,1]) ;nb.xmin <- min(nb.full$maskcoords[,1])
  nb.ymax <- max(nb.full$maskcoords[,2]) ;nb.ymin <- min(nb.full$maskcoords[,2])
  nb.zmax <- max(nb.full$maskcoords[,3]) ;nb.zmin <- min(nb.full$maskcoords[,3])
  #Get re-scaler
  x.scaler <- max(abs(nb.xmax),abs(nb.xmin))
  y.scaler <- max(abs(nb.ymax),abs(nb.ymin))
  z.scaler <- max(abs(nb.zmax),abs(nb.zmin))
  #Re-scale
  nb.full$maskcoords[,1] <- nb.full$maskcoords[,1]/x.scaler
  nb.full$maskcoords[,2] <- nb.full$maskcoords[,2]/y.scaler
  nb.full$maskcoords[,3] <- nb.full$maskcoords[,3]/z.scaler
  
  #get psi for Region 1
  poly_degree = deg_vec
  a_concentration = param_grid[gs.opt[which(1==(res3.mask.reg)),1],2]
  b_smoothness = param_grid[gs.opt[which(1==(res3.mask.reg)),1],3]
  psi.mat.nb <- GP.eigen.funcs.fast(nb.full$maskcoords, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
  bases.nb.to.use <- t(as.matrix(bases.nb))
  #Get psi for other regions  
  for(m in setdiff(res3.mask.reg,1)){
    poly_degree = deg_vec
    a_concentration = param_grid[gs.opt[which(m==(res3.mask.reg)),1],2]
    b_smoothness = param_grid[gs.opt[which(m==(res3.mask.reg)),1],3]
    psi.mat.nb <- GP.eigen.funcs.fast(nb.full$maskcoords, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
    #get lambda
    lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
    #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
    sqrt.lambda.nb <- sqrt(lambda.nb)
    bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
    bases.nb.to.use[m==masked.reg,] <- (t(as.matrix(bases.nb)))[m==masked.reg,]
  }
  
  partial.gp[which(i==res3.mask.reg),,] <- (bases.nb.to.use)
}


gs.opt <- matrix(,ncol=2,nrow = length(res3.mask.reg))
for(i in sort(res3.mask.reg)){
  gs.opt[which(i==(res3.mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/dfn_gs19oct_ROI_",i,".csv")))
}

# bases.nb.temp <- matrix(,ncol=1,nrow= choose(max(deg_vec)+dimension,dimension))



partial.gp <- array(, dim = c(length(res3.mask.reg),p.dat,n.expan))
for(i in res3.mask.reg){
  #Load region mask
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/roi/dfn_mask_ROI_',i))
  invisible(capture.output(nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)))
  #Get the centre of the region
  nb.centre<- apply(nb$maskcoords,2,median)
  #Load full mask
  
  #get coord wrt full mask
  invisible(capture.output(nb.full <- find_brain_image_neighbors(img1, mask_subcor, radius=1)))
  #Re-centre the coords wrt centre of ROI
  nb.full$maskcoords[,1] <- nb.full$maskcoords[,1] - nb.centre[1]
  nb.full$maskcoords[,2] <- nb.full$maskcoords[,2] - nb.centre[2]
  nb.full$maskcoords[,3] <- nb.full$maskcoords[,3] - nb.centre[3]
  #Get max and min coord
  nb.xmax <- max(nb.full$maskcoords[,1]) ;nb.xmin <- min(nb.full$maskcoords[,1])
  nb.ymax <- max(nb.full$maskcoords[,2]) ;nb.ymin <- min(nb.full$maskcoords[,2])
  nb.zmax <- max(nb.full$maskcoords[,3]) ;nb.zmin <- min(nb.full$maskcoords[,3])
  #Get re-scaler
  x.scaler <- max(abs(nb.xmax),abs(nb.xmin))
  y.scaler <- max(abs(nb.ymax),abs(nb.ymin))
  z.scaler <- max(abs(nb.zmax),abs(nb.zmin))
  #Re-scale
  nb.full$maskcoords[,1] <- nb.full$maskcoords[,1]/x.scaler
  nb.full$maskcoords[,2] <- nb.full$maskcoords[,2]/y.scaler
  nb.full$maskcoords[,3] <- nb.full$maskcoords[,3]/z.scaler
  
  #get psi for Region 1
  poly_degree = deg_vec
  a_concentration = param_grid[gs.opt[which(1==(res3.mask.reg)),1],2]
  b_smoothness = param_grid[gs.opt[which(1==(res3.mask.reg)),1],3]
  psi.mat.nb <- GP.eigen.funcs.fast(nb.full$maskcoords, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
  bases.nb.to.use <- t(as.matrix(bases.nb))
  #Get psi for other regions  
  for(m in setdiff(res3.mask.reg,1)){
    poly_degree = deg_vec
    a_concentration = param_grid[gs.opt[which(m==(res3.mask.reg)),1],2]
    b_smoothness = param_grid[gs.opt[which(m==(res3.mask.reg)),1],3]
    psi.mat.nb <- GP.eigen.funcs.fast(nb.full$maskcoords, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
    #get lambda
    lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
    #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
    sqrt.lambda.nb <- sqrt(lambda.nb)
    bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
    bases.nb.to.use[m==masked.reg,] <- (t(as.matrix(bases.nb)))[m==masked.reg,]
  }
  
  partial.gp[which(i==res3.mask.reg),,] <- (bases.nb.to.use)
}
