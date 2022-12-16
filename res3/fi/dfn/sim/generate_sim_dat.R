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

##21 Nov
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov21_small','_inpred__jobid_',1,'.feather'))))
dat.in <- tail(dat.in,30)
dat.in.ex <-  matrix(c(31:1611, rep(0,1581)),ncol=2)
colnames(dat.in) <- c('id','pred')
colnames(dat.in.ex) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
dat.in <- rbind(dat.in,dat.in.ex)
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov21_small','_outpred__jobid_',1,'.feather'))))
dat.out <- tail(dat.out,30)
dat.out.ex <-  matrix(c(1642:3253, rep(0,1611)),ncol=2)
colnames(dat.out) <- c('id','pred')
colnames(dat.out.ex) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
dat.out <- rbind(dat.out,dat.out.ex)
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi_small.csv',row.names = FALSE)


##24 Nov.        THIS IS WRONG ==> IT SHOULD BE NOV24 NOT *NOV14*
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov14_gpnn10_5','_inpred__jobid_',1,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov14_gpnn10_5','_outpred__jobid_',1,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi4.csv',row.names = FALSE)

##25 Nov
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov25_gpnn10_5','_inpred__jobid_',1,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','nov25_gpnn10_5','_outpred__jobid_',1,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi5.csv',row.names = FALSE)


#2 Dec
##Use `fi/pile/re_dec1_gpnnlin_5_loss__jobid_4.csv`
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec1_gpnnlin_5','_inpred__jobid_',4,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec1_gpnnlin_5','_outpred__jobid_',4,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi6.csv',row.names = FALSE)

#4 Dec
##Use `fi/pile/re_dec2_gpnnlin_6_loss__jobid_4.csv`
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec2_gpnnlin_6','_inpred__jobid_',4,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec2_gpnnlin_6','_outpred__jobid_',4,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi7.csv',row.names = FALSE)
# prior.var <- 0.05
# l.prior.var <- 5e-6
# learning_rate <- 5e-11 #for slow decay starting less than 1
# prior.var.bias <- 1
# epoch <- 1000

#7 Dec
##After Jian's comment, add noise to 
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec2_gpnnlin_6','_inpred__jobid_',4,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec2_gpnnlin_6','_outpred__jobid_',4,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
set.seed(1)
dat.sim$pred <- dat.sim$pred + rnorm(length(dat.sim$pred),0,3)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi8.csv',row.names = FALSE)
set.seed(NULL)


#9 Dec
##Generating from `gen_pred_gpr_ldorsal`
ind.to.use<- list()
ind.to.use$test <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index.csv')$x
ind.to.use$train <-  read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index.csv')$x
##In sample
dat.in <- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_dec7_gpr_LDor_inpred_noscale_4.csv')$x
dat.in <- cbind(ind.to.use$train,dat.in)
colnames(dat.in) <- c('id','pred')
##Out of sample
dat.out <- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_dec7_gpr_LDor_outpred_noscale_4.csv')$x
dat.out <- cbind(ind.to.use$test,dat.out)
colnames(dat.out) <- c('id','pred')
##Merge
dat.sim <- rbind(dat.in,dat.out)
# set.seed(1)
# dat.sim$pred <- dat.sim$pred + rnorm(length(dat.sim$pred),0,3)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi_ldor.csv',row.names = FALSE)
set.seed(NULL)

#15 Dec
##Generate data from `gen_pred_ols` and add noise
##In sample
dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec14_gpnn_ols_gen','_inpred__jobid_',4,'.feather'))))
dat.in <- tail(dat.in,1611)
colnames(dat.in) <- c('id','pred')
dat.in <- dat.in[order(dat.in$id),]
##Out of sample
dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_','dec14_gpnn_ols_gen','_outpred__jobid_',4,'.feather'))))
dat.out <- tail(dat.out,1642)
colnames(dat.out) <- c('id','pred')
dat.out <- dat.out[order(dat.out$id),]
##Merge
dat.sim <- rbind(dat.in,dat.out)
set.seed(1)
dat.sim$pred <- dat.sim$pred + rnorm(length(dat.sim$pred),0,1)
write.csv(dat.sim,'/well/nichols/users/qcv214/bnn2/res3/fi/dfn/sim/simfi_ols.csv',row.names = FALSE)
set.seed(NULL)

