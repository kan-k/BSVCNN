#For creating partial GP
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

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.reg <- sort(setdiff(unique(c(mask_subcor)),c(0)))

#load preset
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))

# norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }

dimension = 3
# deg_vec<- c(30,40)
# a_vec  <- c(0.001,0.1,0.5,1,2,3)
# b_vec  <- c(1,5,10,20,40,60,80,100,120,140)
# param_grid <- as.matrix(expand.grid(deg_vec,a_vec,b_vec))
# gs.opt <- matrix(,ncol=2,nrow = length(mask.reg))
# for(i in sort(mask.reg)){
#   # gs.opt[which(i==(mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/pile/gs23feb_ROI_",i,".csv")))
#   gs.opt[which(i==(mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/pile/gs25feb_ROI_",i,".csv")))
# }

# bases.nb.temp <- matrix(,ncol=1,nrow= choose(max(deg_vec)+dimension,dimension))

for(i in JobId){
  # poly_degree = param_grid[gs.opt[which(i==(mask.reg)),1],1] #######I forgot that it must the same poly degree....
  # a_concentration= param_grid[gs.opt[which(i==(mask.reg)),1],2]
  # b_smoothness= param_grid[gs.opt[which(i==(mask.reg)),1],3]
  poly_degree = 10
  a_concentration = 0.5
  b_smoothness = 40
  
  #Load region mask
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/roi/mask_ROI_',i))
  nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)
  #Get the centre of the region
  nb.centre<- apply(nb$maskcoords,2,median)
  #Load full mask
  mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
  #get coord wrt full mask
  nb.full <- find_brain_image_neighbors(img1, mask_subcor, radius=1)
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

  #get psi
  psi.mat.nb <- GP.eigen.funcs.fast(nb.full$maskcoords, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
  # if(nrow(bases.nb) < choose(max(deg_vec)+dimension,dimension)){
  #   empty.expansion <- matrix(0, nrow= choose(max(deg_vec)+dimension,dimension)-nrow(bases.nb), ncol= ncol(bases.nb))
  #   bases.nb <- rbind(bases.nb,empty.expansion)
  # }
  #print(paste0("here, ",i))
  # bases.nb.temp <- cbind(as.matrix(bases.nb.temp),as.matrix(bases.nb))
  bases.nb <- as.matrix(bases.nb)
  write_feather(as.data.frame(bases.nb),paste0("/well/nichols/users/qcv214/bnn2/res3/roi/partial_gp_",i,"_fixed_",poly_degree,a_concentration,b_smoothness,".feather"))
}
# colnames(bases.nb.temp) <- NULL
# bases.nb <- as.data.frame(bases.nb.temp[,-1])
# bases.nb <- as.matrix(bases.nb)


# print("before cbind")
#Get design matrix
# z.nb <- cbind(1,t(bases.nb%*%t(dat_allmat)))

