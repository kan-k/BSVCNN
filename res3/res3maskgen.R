if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

temp.img <-oro.nifti::readNIfTI('/well/nichols/users/kfh142/data/Atlas/CIC/MultRes/CICatlas_Res3_2mm.nii.gz')
#Take only grey matter
temp.img[temp.img>12] <- 0
temp.img@datatype = 2
temp.img@bitpix = 8
writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/maskfix/res3mask_1'))
#Load data with new unthreshold mask
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
res3.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/maskfix/res3mask_1') 
#threshold at 0.1
dat_colmeans<-colMeans(res3.dat)
#Originally I had temp.img > 1, that means it's not thresholded by 
# temp.img[temp.img>1][dat_colmeans<0.1] <- 0 
# temp.img@datatype = 2
# temp.img@bitpix = 8
# writeNIfTI(temp.img,'/well/nichols/users/qcv214/bnn2/res3/maskfix/res3mask_2')

#
temp.img[temp.img>0][dat_colmeans<0.1] <- 0 
temp.img@datatype = 2
temp.img@bitpix = 8
# writeNIfTI(temp.img,'/well/nichols/users/qcv214/bnn2/res3/maskfix/res3mask_3') #Correct 
writeNIfTI(temp.img,'/well/nichols/users/qcv214/bnn2/res3/res3mask')

#Load data again using the new thresholded mask
res3.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask') # 4263 x 156812
#save datat
# write_feather(as.data.frame(res3.dat), '/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather') #####Re-change
#reload data and mask
res3.dat <- read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather') #dim = 4263 156812
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
res3.mask.reg <- setdiff(sort(unique(c(res3.mask))),0)

print(res3.mask.reg)
print(dim(res3.dat))




