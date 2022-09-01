##Simulation study
# Sub-sample res4 by 1/10
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)

# res4.dat <- read_feather('/well/nichols/users/qcv214/bnn2/res3/res4_dat.feather') #dim = 4263 124859
res4.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz')
res4.mask.reg <- setdiff(sort(unique(c(res4.mask))),0)

# masked.index <- seq_along(res4.mask[res4.mask>0])
# res4.mask[res4.mask>0][(masked.index%%10)!=0] <- 0
# res4.mask@datatype = 2
# res4.mask@bitpix = 8
# writeNIfTI(res4.mask,paste0('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub'))
# 
# sub.res4.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
# sub.res4.mask.reg <- setdiff(sort(unique(c(res4.mask))),0)
# part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
# part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# # 
# list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.res4.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub') # 4263 x 156812
# #save datat
# write_feather(as.data.frame(sub.res4.dat), '/well/nichols/users/qcv214/bnn2/res3/sim/sub_res4_dat.feather')


#####Found out on FSLeyes that res3 and res4 don't overlap in some ways... I think I will just use non-zero res3 and input res4 value 
#Use fast_read_imgs_mask, use full res4 as data, and use res3_sub as mask
# mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res3_sub.nii.gz')
sub.res4.2 <- fast_read_imgs_mask('/well/nichols/users/qcv214/bnn2/res3/res4mask.nii.gz','/well/nichols/users/qcv214/bnn2/res3/sim/res3_sub.nii.gz')
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res3_sub.nii.gz')
gp.mask.hs <- res3.mask
gp.mask.hs[gp.mask.hs!=0] <- c(sub.res4.2)
gp.mask.hs@datatype = 2
gp.mask.hs@bitpix = 8
writeNIfTI(gp.mask.hs,paste0('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub'))

sub.res4.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
sub.res4.mask.reg <- setdiff(sort(unique(c(res4.mask))),0)
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
sub.res4.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub') # 4263 x 156812
#save datat
write_feather(as.data.frame(sub.res4.dat), '/well/nichols/users/qcv214/bnn2/res3/sim/sub_res4_dat.feather')

#### Generate region ROI
#### SEE res4_roi.R
# part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
# part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# #These two are equal
# part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# 
# res4.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/sim/sub_res4_dat.feather'))
# res4.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
# res4.mask.reg <- setdiff(sort(unique(c(res4.mask))),0)
# 
# for(roi in res4.mask.reg){
#   print(roi)
#   temp.img <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
#   temp.img <- temp.img == roi #1 to 55
#   temp.img@datatype = 2
#   temp.img@bitpix = 8
#   writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_mask_ROI_',roi))
#   temp.dat <- fast_read_imgs_mask(list_of_all_images,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_mask_ROI_',roi)) # 4263 x 163375
#   write_feather(as.data.frame(temp.dat), paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_dat_ROI_',roi,'.feather'))
# }





