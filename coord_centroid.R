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
nb.full <- find_brain_image_neighbors(img1, mask_subcor, radius=1)
nb.xmax <- max(nb.full$maskcoords[,1]) ;nb.xmin <- min(nb.full$maskcoords[,1])
nb.ymax <- max(nb.full$maskcoords[,2]) ;nb.ymin <- min(nb.full$maskcoords[,2])
nb.zmax <- max(nb.full$maskcoords[,3]) ;nb.zmin <- min(nb.full$maskcoords[,3])
norm.func.full <- function(x){
  x.out <- 2*(x[1] - nb.xmin)/(nb.xmax-nb.xmin) -1
  y.out <- 2*(x[2] - nb.ymin)/(nb.ymax-nb.ymin) -1
  z.out <- 2*(x[3] - nb.zmin)/(nb.zmax-nb.zmin) -1
  return(c(x.out,y.out,z.out))
}
#coord.full <- apply(nb.full$maskcoords[1:3,],2,norm.func)

centroid <- matrix(,nrow=length(mask.reg),ncol = 7)
for(i in mask.reg){
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bsvcnn/pile/combined/fin_mask_ROI_',i))
  nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)
  nb.centred <- apply(nb$maskcoords,2,norm.func)
  centroid.temp <- apply(nb$maskcoords,2,median)
  
  centroid[which(i==mask.reg),] <- c(i, centroid.temp, norm.func.full(centroid.temp)) 
}
#Save centroid coord
write.csv(centroid,paste0("/well/nichols/users/qcv214/bnn2/pile/coord_centroids.csv"), row.names = FALSE)
print("done")
