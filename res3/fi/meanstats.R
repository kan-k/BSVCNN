#Function for taking in multiple runs' predictions, and calculate MAE, MSE and R^2
if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(dplyr)

#calculating r^2
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return((1 - RSS/TSS)*100)
}

meanstats.re<- function(filename, runs){
  in.rmse.vec<- vector(mode="numeric")
  in.mae.vec<- vector(mode="numeric")
  in.rsq.vec<- vector(mode="numeric")
  out.rmse.vec<- vector(mode="numeric")
  out.mae.vec<- vector(mode="numeric")
  out.rsq.vec<- vector(mode="numeric")
  
  num.run <- length(runs)
  for(i in runs){
    #Load in-sample
    dat.in <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_',filename,'_inpred__jobid_',i,'.feather'))))
    dat.in <- tail(dat.in,1611)
    colnames(dat.in) <- c('id','pred')
    age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/fi.feather')

####MODIFYTHS####################################################################################################
    age_tab <- age_tab[order(age_tab$id),]
    ####################################################################################################    
    
    wb2.train.id<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/train_index.csv')$x
    wb2.train.pred <- age_tab$fi[wb2.train.id]
    true.train<- data.frame(wb2.train.id,wb2.train.pred,row.names = NULL)
    colnames(true.train) <- c('id','true')
    dat.in <- merge(dat.in,true.train, by = 'id')
    ##Metrics
    in.rmse <- (mean((dat.in$pred-dat.in$true)^2))
    in.mae <- (mean(abs(dat.in$pred-dat.in$true)))
    in.rsq <- rsqcal(dat.in$true,dat.in$pred)
    
    #Load out-of-sample
    dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_',filename,'_outpred__jobid_',i,'.feather'))))
    dat.out <- tail(dat.out,1642)
    colnames(dat.out) <- c('id','pred')
    wb2.test.id<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/dfn/test_index.csv')$x
    wb2.test.pred <- age_tab$fi[wb2.test.id]
    true.test<- data.frame(wb2.test.id,wb2.test.pred,row.names = NULL)
    colnames(true.test) <- c('id','true')
    dat.out <- merge(dat.out,true.test, by = 'id')
    ##Metrics
    out.rmse <- (mean((dat.out$pred-dat.out$true)^2))
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
  # out[1,] <- c(round(mean(in.rmse.vec),2),round(sd(in.rmse.vec),3))
  # out[3,] <- c(round(mean(in.mae.vec),2),round(sd(in.mae.vec),3))
  # out[2,] <- c(round(mean(in.rsq.vec),2),round(sd(in.rsq.vec),3))
  # out[4,] <- c(round(mean(out.rmse.vec),2),round(sd(out.rmse.vec),3))
  # out[6,] <- c(round(mean(out.mae.vec),2),round(sd(out.mae.vec),3))
  # out[5,] <- c(round(mean(out.rsq.vec),2),round(sd(out.rsq.vec),3))
  out[1,] <- c(round(sqrt(median(in.rmse.vec)),2),round(sqrt(sd(in.rmse.vec)),3))
  out[3,] <- c(round(median(in.mae.vec),2),round(sd(in.mae.vec),3))
  out[2,] <- c(round(median(in.rsq.vec),2),round(sd(in.rsq.vec),3))
  out[4,] <- c(round(sqrt(median(out.rmse.vec)),2),round(sqrt(sd(out.rmse.vec)),3))
  out[6,] <- c(round(median(out.mae.vec),2),round(sd(out.mae.vec),3))
  out[5,] <- c(round(median(out.rsq.vec),2),round(sd(out.rsq.vec),3))
  out <-  as.data.frame(out)
  colnames(out) <- c("mean", "sd")
  rownames(out) <- c("inRMSE","inR2","inMAE","outRMSE","outR2","outMAE")
  return(out)
}
date <- "nov6"

stat.vanilla <- meanstats.re("nov2_sub_gpnn_lr00005",c(14,17,2,22,3,7,9))
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/",date,"_re_median_summary_gpnn_lr00005.csv"))

stat.vanilla <- meanstats.re("nov4_gpnnhc", c(12,14,17,2,6,9))
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/",date,"_re_median_summary_gpnnhc.csv"))

stat.vanilla <- meanstats.re("nov2_sub_stgp_th25",1:30)
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/",date,"_re_median_summary_stgp_th25.csv"))

stat.vanilla <- meanstats.re("nov2_stgp_th100",1:30)
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/",date,"_re_median_summary_stgp_th100.csv"))




