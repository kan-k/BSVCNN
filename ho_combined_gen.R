#Produce combined Harvard Oxford Subcortical mask

#Load package
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(oro.nifti)
p_load(neurobase)

#Load HO Cortical from fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz
cort.og <- oro.nifti::readNIfTI('/well/win/software/packages/fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz')
#Load HO Sub-Cortical from fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz
subcort.og <- oro.nifti::readNIfTI('/well/win/software/packages/fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz')

#Update -- I should threshold cortical from subcortical first, ie remove white matters and brain stem
# cort.og[c(subcort.og) == 1] <- 0
# cort.og[c(subcort.og) == 8] <- 0
# cort.og[c(subcort.og) == 12] <- 0
#
#2nd attempt tp update
subcort.og[subcort.og == 1] <- 0
subcort.og[subcort.og == 8] <- 0
subcort.og[subcort.og == 12] <- 0
# writeNIfTI(subcort.og,paste0('/well/nichols/users/qcv214/bnn2/HO_Combined/subcort.wm.removed'))

cort.og[subcort.og==0] <- 0
# cort.og@datatype = 2
# cort.og@bitpix = 8
# writeNIfTI(cort.og,paste0('/well/nichols/users/qcv214/bnn2/HO_Combined/cort.filtered'))


cort.array <- array(c(cort.og), dim = dim(cort.og))
mid.x.brain <- (ceiling(dim(cort.array)[1]/2))
max.x.brain <- dim(cort.array)[1]
cort.array[mid.x.brain:max.x.brain,,] <- cort.array[mid.x.brain:max.x.brain,,] + max(sort(unique(c(cort.array)))) + 1
cort.array[c(cort.array)==49] <- 0
cort.array[mid.x.brain:max.x.brain,,] <- cort.array[mid.x.brain:max.x.brain,,] - 1
cort.array[c(cort.array)== -1] <- 0

cort.og[cort.og>0] <- cort.array[cort.og > 0]
writeNIfTI(cort.og,paste0('/well/nichols/users/qcv214/bnn2/HO_Combined/cort.arr'))
# cort.temp <-oro.nifti::readNIfTI('/well/win/software/packages/fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz')
# cort.temp[cort.temp!=0] <- cort.array[cort.array!=0]
cort.temp <- cort.og

#Add in sub-cortical labels
cort.temp[c(cort.temp)==0] <- subcort.og[c(cort.temp)==0] + max(c(cort.temp)) + 1
cort.temp[c(cort.temp)==97] <- 0
cort.temp[c(cort.temp)>97] <- cort.temp[c(cort.temp)>97] - 1
cort.temp[c(cort.temp)== -1] <- 0

#Remove 
# cort.temp[c(cort.temp)==97] <- 0 #Remove left Cerebral White Matter
# cort.temp[c(cort.temp)==97+7] <- 0 #Remove brainstem
# cort.temp[c(cort.temp)==97+11] <- 0 #Remove right Cerebral White Matter
cort.temp[c(cort.temp)==97+6] <- 0 #Remove left Pallidum
cort.temp[c(cort.temp)==97+17] <- 0 #Remove right Pallidum

#Overlapping clean-up
cort.temp[c(cort.temp)==98] <- 0 
cort.temp[c(cort.temp)==109] <- 0 

cort.temp@datatype = 2
cort.temp@bitpix = 8
writeNIfTI(cort.temp,paste0('/well/nichols/users/qcv214/bnn2/HO_Combined/cort.temp'))

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1:(dim(part_use)[1]),1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
ho.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/HO_Combined/cort.temp') 
#threshold at 0.1
dat_colmeans<-colMeans(ho.dat)
cort.temp[cort.temp>0][dat_colmeans<0.1] <- 0 
cort.temp@datatype = 2
cort.temp@bitpix = 8
writeNIfTI(cort.temp,'/well/nichols/users/qcv214/bnn2/HO_Combined/thres') #Correct
