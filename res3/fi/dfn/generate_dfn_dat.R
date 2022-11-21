#Generate default mode network data

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_list$exist_fmri <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))
#There are fewer participants with fmri 

# sum(part_list$exist_vbm)
# [1] 4262

# sum(part_list$exist_fmri)
# [1] 3735
#There are fewer participants with fmri 

part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# sum(part_use$exist_fmri)
# [1] 3735
#All fmri have vbm


#Read in train-test index corresponding to part_use
train.test.ind <- list()
train.test.ind$test <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/train_index.csv')$x

part.use.train <-part_use[c(train.test.ind$train),]
part.use.train <-part.use.train[part.use.train$exist_fmri==1,]
# sum(part.use.train$exist_fmri)
# [1] 1611
part.use.test <-part_use[c(train.test.ind$test),]
part.use.test <-part.use.test[part.use.test$exist_fmri==1,]
# sum(part.use.test$exist_fmri)
# [1] 1642


list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part.use.train[1:2,1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz')
dfn.res3.dat <- fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask') 
# dim(dfn.res3.dat)
# [1]      2 156812

#Try manually load first participant
mask.com<- oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part.use.train[1,1],'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))
img1 <- (img1[,,,1])[mask.com>0]

#Seems like `fast_read_imgs_mask` reads the first image

#Concat the list of usable participants
part.full <- c(part.use.train$V1,part.use.test$V1)
write.csv(part.full,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv', row.names = FALSE)

part.full<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x

write.csv(1:1611, '/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index.csv', row.names = FALSE)
write.csv(1612:3253, '/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index.csv', row.names = FALSE)


train.test.ind <- list()
train.test.ind$test <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index.csv')$x

part.full



#Write out new FI
fitab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
fi_tab<-as.data.frame(matrix(,nrow = length(part.full$x),ncol = 2)) #id, fi, number of masked voxels
colnames(fi_tab)[1:2]<-c('id','fi') 
fi_tab$id<-part.full$x
for(i in 1:length(part.full$x)){
  fi_tab$fi[i]<-fitab$X20016.2.0[fitab$eid_8107==sub(".", "",fi_tab$id[i])] #change from age $X21003.2.0 to Fluid Intelligence $X20016.2.0
}

write_feather(fi_tab, '/well/nichols/users/qcv214/bnn2/res3/fi/dfn/fi.feather')




############################After 7 Nov,  data went missing
# [1] "Missing total: 335"
# [1] "Missing train: 162"
# [1] "Missing test: 173"
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
part.exist <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use,'/fMRI/rfMRI_25.dr/dr_stage2.nii.gz'))
write.csv(part_use[part.exist],'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id2.csv', row.names = FALSE)
write.csv(1:1449, '/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index2.csv', row.names = FALSE)
write.csv(1450:2918, '/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index2.csv', row.names = FALSE)
############################9 Nov, above was tempaorary
