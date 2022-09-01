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

JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')

res4.dat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/sim/sub_res4_dat.feather'))
res4.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
res4.mask.reg <- setdiff(sort(unique(c(res4.mask))),0)

temp.img <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/sim/res4_sub.nii.gz')
temp.img <- temp.img == JobId #1 to 55
temp.img@datatype = 2
temp.img@bitpix = 8
writeNIfTI(temp.img,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_mask_ROI_',JobId))
temp.dat <- fast_read_imgs_mask(list_of_all_images,paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_mask_ROI_',JobId)) # 4263 x 163375
write_feather(as.data.frame(temp.dat), paste0('/well/nichols/users/qcv214/bnn2/res3/roi/sim_res4_dat_ROI_',JobId,'.feather'))