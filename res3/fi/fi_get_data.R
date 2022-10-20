#To extract Fluid Intelligence
#Refer to the code /well/nichols/users/qcv214/ image_extract.Rmd  or image_extract_script.R 

library(PMS)
library(oro.nifti)
library(neurobase)
agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)



#11oct
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left

#Look at intersection, my current id on participant list agrees with eid_8107 NOT normal eid
length(intersect(part_use$V1,paste0(2,agetab$eid_8107)))

#Taken from PLcaement_2_V2
fitab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
fi_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, fi, number of masked voxels
colnames(fi_tab)[1:2]<-c('id','fi') 
fi_tab$id<-part_use$V1
for(i in 1:length(part_use$V1)){
  fi_tab$fi[i]<-fitab$X20016.2.0[fitab$eid_8107==sub(".", "",fi_tab$id[i])] #change from age $X21003.2.0 to Fluid Intelligence $X20016.2.0
}


summary(fi_tab$fi) #There are 319 NAs

# plot(hist(age_tab$age[!is.na(age_tab$age)]),main='Fluid Intelligence, n = 3,943',xlab='Fluid Intelligence score')

#Note that the id in `fi_tab` and `age_tab` coincides, so can split with my train test split:
ind.temp <- read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv"))
train.test.ind <- list()
train.test.ind$test <- unlist(ind.temp[2,])
train.test.ind$train <- unlist(ind.temp[1,])

#Check how many NAs are in train and test
print(paste0("Train no.NAs: ", sum(is.na(fi_tab$fi[train.test.ind$train]))))
print(paste0("Test no.NAs: ", sum(is.na(fi_tab$fi[train.test.ind$test]))))

#Remove NA from the train and test split
train.test.ind.train <- train.test.ind$train[!is.na(fi_tab$fi[train.test.ind$train])]
train.test.ind.test <- train.test.ind$test[!is.na(fi_tab$fi[train.test.ind$test])]

#Save fi data AND the new train and test split
write_feather(fi_tab, '/well/nichols/users/qcv214/bnn2/res3/fi/fi.feather')
write.csv(train.test.ind.train, '/well/nichols/users/qcv214/bnn2/res3/fi/train_index.csv')
write.csv(train.test.ind.test, '/well/nichols/users/qcv214/bnn2/res3/fi/test_index.csv')


#Perhaps modify algorithm to just take sampling per iteration.




