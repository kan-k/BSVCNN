#RScript

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

#Create a threshold mask
temp.img <-oro.nifti::readNIfTI('/well/nichols/users/kfh142/data/Atlas/CIC/MultRes/CICatlas_Res3_2mm.nii.gz')
# #Take only grey matter
temp.img[temp.img>12] <- 0
temp.img@datatype = 2
temp.img@bitpix = 8
writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/res3mask_nowm'))
#Load data with new unthreshold mask
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use,'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz')
res3.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask_nowm.nii.gz')
#threshold at 0.1
# dat_colmeans<-colMeans(res3.dat)
# temp.img[temp.img>0][abs(dat_colmeans)<0.5] <- 0  #threshold of 0.5
dat_colsd <- apply(res3.dat, 2, sd)
temp.img[temp.img>0][dat_colsd<1.5] <- 0  #threshold of 0.5
temp.img@datatype = 2
temp.img@bitpix = 8
writeNIfTI(temp.img,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/res3mask')
#Load data again using the new thresholded mask
# res3.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask') #
#save datat
# write_feather(as.data.frame(res3.dat), '/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather')
#reload data and mask
# res3.dat <- read_feather('/well/nichols/users/qcv214/bnn2/res3/res3_dat.feather') #dim = 4263 124859
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/res3mask.nii.gz')
res3.mask.reg <- setdiff(sort(unique(c(res3.mask))),0)


#Do ROI
res3.mask.reg <- sort(unique(c(res3.mask)))
for(i in res3.mask.reg){
  temp.img <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/res3mask.nii.gz')
  temp.img <- temp.img == i 
  temp.img@datatype = 2
  temp.img@bitpix = 8
  writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/dfn_mask_ROI_',i))
  # temp.dat <- fast_read_imgs_mask(list_of_all_images,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/dfn_mask_ROI_',i)) # 4263 x 163375
  # write_feather(as.data.frame(temp.dat), paste0('/well/nichols/users/qcv214/bnn2/res3/roi/dat_ROI_',i))
}