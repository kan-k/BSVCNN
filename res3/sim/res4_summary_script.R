if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(dplyr)

rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return(1 - RSS/TSS)
}


meanstats.sim<- function(filename, runs){
  # path <- paste0('/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_loss__jobid_',runs,'.csv')
  in.rmse.vec<- vector(mode="numeric")
  in.mae.vec<- vector(mode="numeric")
  in.rsq.vec<- vector(mode="numeric")
  out.rmse.vec<- vector(mode="numeric")
  out.mae.vec<- vector(mode="numeric")
  out.rsq.vec<- vector(mode="numeric")
  
  num.run <- length(runs)
  for(i in runs){
    print(paste0("Doing run: ",i))
    #Load in-sample
    dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_inpred__jobid_',i,'.feather'))))
    dat.in <- tail(dat.in,2000)
    colnames(dat.in) <- c('id','pred')
    wb2.train.id<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv")))[1,]
    wb2.train.pred <- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_inpred_4_addednoise.csv")))
    true.train<- data.frame(wb2.train.id,wb2.train.pred,row.names = NULL)
    colnames(true.train) <- c('id','true')
    dat.in <- merge(dat.in,true.train, by = 'id')
    ##Metrics
    in.rmse <- sqrt(mean((dat.in$pred-dat.in$true)^2))
    in.mae <- (mean(abs(dat.in$pred-dat.in$true)))
    in.rsq <- rsqcal(dat.in$true,dat.in$pred)
    
    #Load out-of-sample
    dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/pile/sim_',filename,'_outpred__jobid_',i,'.feather'))))
    dat.out <- tail(dat.out,2000)
    colnames(dat.out) <- c('id','pred')
    wb2.test.id<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_index_",4,".csv")))[2,]
    wb2.test.pred <- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_wb2_outpred_4_addednoise.csv")))
    true.test<- data.frame(wb2.test.id,wb2.test.pred,row.names = NULL)
    colnames(true.test) <- c('id','true')
    dat.out <- merge(dat.out,true.test, by = 'id')
    ##Metrics
    out.rmse <- sqrt(mean((dat.out$pred-dat.out$true)^2))
    out.mae <- (mean(abs(dat.out$pred-dat.out$true)))
    out.rsq <- rsqcal(dat.out$true,dat.out$pred)
    
    #concat result
    in.rmse.vec <- c(in.rmse.vec,in.rmse)
    in.mae.vec <- c(in.mae.vec,in.mae)
    in.rsq.vec <- c(in.rsq.vec,in.rsq)
    
    out.rmse.vec <- c(out.rmse.vec,out.rmse)
    out.mae.vec <- c(out.mae.vec,out.mae)
    out.rsq.vec <- c(out.rsq.vec,out.rsq)
  }
  out <- matrix(,nrow=6,ncol=2)
  out[1,] <- c(mean(in.rmse.vec),sd(in.rmse.vec))
  out[2,] <- c(mean(in.mae.vec),sd(in.mae.vec))
  out[3,] <- c(mean(in.rsq.vec),sd(in.rsq.vec))
  out[4,] <- c(mean(out.rmse.vec),sd(out.rmse.vec))
  out[5,] <- c(mean(out.mae.vec),sd(out.mae.vec))
  out[6,] <- c(mean(out.rsq.vec),sd(out.rsq.vec))
  out <-  as.data.frame(out)
  colnames(out) <- c("mean", "sd")
  rownames(out) <- c("inRMSE","inMAE","inR2","outRMSE","outMAE","outR2")
  return(out)
}



print("start=========")
stat.ig <- meanstats.sim("res4_aug17_nnvwbig_a6",c(10,11,12,13,15,17,2,21,25,26,27,28,29,3,30,31,33,37,38,4,43,44,45,47,48,49,5,50,6,7,8))
write.csv(stat.ig,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_res4_summary_ig.csv"))
print("=========IG DONE=========")

stat.eb <-meanstats.sim("res4_aug17_nnvwbbayes_initvar", c(1,10,11,12,13,15,16,17,19,2,20,23,24,29,3,30,33,34,37,39,4,40,42,44,48,5,50,6,8))
write.csv(stat.eb,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_res4_summary_eb.csv"))
print("=========EB DONE=========")
stat.vanilla <- meanstats.sim("res4_aug17_nnvwb4", c(10,12,15,17,2,21,24,25,26,27,29,3,30,33,34,35,36,38,41,44,46,47,48,5,50,6,7,9))
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/sim_res4_summary_van.csv"))

print(stat.ig)
print(stat.eb)
print(stat.vanilla)





