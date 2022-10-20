#Function for taking in multiple runs' predictions, and calculate MAE, MSE and R^2

#calculating r^2
rsqcal <- function(true,pred){
  RSS <-sum((true - pred)^2)
  TSS <- sum((true - mean(true))^2)
  return(1 - RSS/TSS)
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
    dat.in <- tail(dat.in,1839)
    colnames(dat.in) <- c('id','pred')
    age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/fi/fi.feather')
    wb2.train.id<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/train_index.csv')$x
    wb2.train.pred <- age_tab$fi[wb2.train.id]
    true.train<- data.frame(wb2.train.id,wb2.train.pred,row.names = NULL)
    colnames(true.train) <- c('id','true')
    dat.in <- merge(dat.in,true.train, by = 'id')
    ##Metrics
    in.rmse <- sqrt(mean((dat.in$pred-dat.in$true)^2))
    in.mae <- (mean(abs(dat.in$pred-dat.in$true)))
    in.rsq <- rsqcal(dat.in$true,dat.in$pred)
    
    #Load out-of-sample
    dat.out <- as.data.frame(t(read_feather(paste0('/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_',filename,'_outpred__jobid_',i,'.feather'))))
    dat.out <- tail(dat.out,1839)
    colnames(dat.out) <- c('id','pred')
    wb2.test.id<- read.csv('/well/nichols/users/qcv214/bnn2/res3/fi/test_index.csv')$x
    wb2.test.pred <- age_tab$fi[wb2.test.id]
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

stat.vanilla <- meanstats.re("oct15_gpnn_lr00005",c(22,28))
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_summary_gpnn_lr00005.csv"))

stat.vanilla <- meanstats.re("oct15_gpnn_lr00005_init",c(12,15,21,24,6))
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_summary_gpnn_lr00005_init.csv"))

stat.vanilla <- meanstats.re("oct14_stgp_lr00005",1:10)
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_summary_stgp_lr00005.csv"))

stat.vanilla <- meanstats.re("oct14_stgp_lr00005_init",1:10)
print("=========van DONE=========")
write.csv(stat.vanilla,paste0("/well/nichols/users/qcv214/bnn2/res3/fi/pile/re_summary_stgp_lr00005_init.csv"))





