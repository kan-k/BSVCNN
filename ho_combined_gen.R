#Produce combined Harvard Oxford Subcortical mask

#Load package
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(oro.nifti)
p_load(neurobase)

#Load HO Cortical from fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz
cort.og <- oro.nifti::readNIfTI('/well/win/software/packages/fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz')
cort.array <- array(c(cort.og), dim = dim(cort.og))
#Load HO Sub-Cortical from fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz
subcort.og <- oro.nifti::readNIfTI('/well/win/software/packages/fsl/6.0.3/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz')

mid.x.brain <- (ceiling(dim(cort.array)[1]/2))
max.x.brain <- dim(cort.array)[1]
cort.array[mid.x.brain:max.x.brain,,] <- cort.array[mid.x.brain:max.x.brain,,] + max(sort(unique(c(cort.array)))) + 1
cort.array[c(cort.array)==49] <- 0
cort.array[mid.x.brain:max.x.brain,,] <- cort.array[mid.x.brain:max.x.brain,,] - 1
cort.array[c(cort.array)== -1] <- 0

cort.array[c(cort.array)==0] <- subcort.og[c(cort.array)==0] + max(c(cort.array)) + 1
cort.array[c(cort.array)==97] <- 0
cort.array[c(cort.array)>97] <- cort.array[c(cort.array)>97] - 1
cort.array[c(cort.array)== -1] <- 0




