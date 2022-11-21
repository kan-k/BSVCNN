#Get simulation response for NN based on FI predictions

#Use /well/nichols/users/qcv214/bnn2/res3/fi/pile/re_nov11_gpnn10_5_loss__jobid_1

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(dplyr)


##
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x


##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov11_gpnn10_5','_inpred__jobid_',1,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov11_gpnn10_5','_outpred__jobid_',1,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]

##Merge
dat.sim <- rbind(dat.in,dat.out)

write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi.csv',row.names = FALSE)
##Modelling the above isn't successful, try with var = 6


#17 Nov
##Using var = 6 magnitude
part_use<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/part_id.csv')$x
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov11_gpnn10_6','_inpred__jobid_',4,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov11_gpnn10_6','_outpred__jobid_',4,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi2.csv',row.names = FALSE)

##18 Nov
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov17_gpnn10_5','_inpred__jobid_',1,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov17_gpnn10_5','_outpred__jobid_',1,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]

##Merge
dat.sim <- rbind(dat.in,dat.out)

write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi3.csv',row.names = FALSE)


