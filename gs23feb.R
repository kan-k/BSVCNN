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

print("stage 1")
JobId=as.numeric(Sys.getenv("SGE_TASK_ID"))

mask_subcor<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bsvcnn/pile/HO-combined-fin.nii.gz')
mask.reg <- sort(setdiff(unique(c(mask_subcor)),c(0)))
mask.reg.part<-split(mask.reg, sort(mask.reg%%100))

#load preset
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))


#load age
agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
for(i in 1:length(part_use$V1)){
  age_tab$age[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab$id[i])]
}
dat.age <-age_tab$age

#func
get_ind <- function(num.train,num.test = 1){
  full.ind<-1:dim(dat_allmat)[1]
  train<-sample(x = full.ind,size = num.train,replace = FALSE)
  test<-sample(x=setdiff(full.ind,train),size=num.test,replace=FALSE)
  out=list()
  out$train<-train
  out$test<-test
  return(out)
}
rsqcal2<-function(old,new,ind.old,ind.new){
  ridge_pred<-old
  ridge_pred_new<-new
  no<-length(old)
  nn<-length(new)
  #insample
  y<-age_tab$age[ind.old]
  sserr_ridge<-sum((y-ridge_pred)^2)
  sstot<-var(y)*length(y)
  #outsample
  y_new<-age_tab$age[ind.new]
  sstot_new<-sum((y_new-mean(y))^2)
  sserr_ridge_new<-sum((y_new-ridge_pred_new)^2)
  #Output
  #print(paste0('In-sameple Variance of prediction explained: ',round(1-sserr_ridge/sstot,5)*100,' || RMSE: ', round(sqrt(mean((y-ridge_pred)^2)),4)))
  #print(paste0('Out-sample Variance of prediction explained: ',round(1-sserr_ridge_new/sstot_new,5)*100,' || RMSE: ', round(sqrt(mean((y_new-ridge_pred_new)^2)),4)))
  print('Done')
  out=list()
  out$inrsq<-round(1-sserr_ridge/sstot,5)*100
  out$inmae<-round(median(abs(y-ridge_pred)),4)
  out$outrsq<-round(1-sserr_ridge_new/sstot_new,5)*100
  out$outmae<-round(median(abs(y_new-ridge_pred_new)),4)
  out$inrmse <-round(sqrt(mean((y-ridge_pred)^2)),4)
  out$outrmse <-round(sqrt(mean((y_new-ridge_pred_new)^2)),4)
  return(out)
}
norm.func <- function(x){ 2*(x - min(x))/(max(x)-min(x)) -1 }





for(k in mask.reg.part[[JobId]]){
  
  #Load rearranged data
  dat_allmat <- read_feather(paste0('/well/nichols/users/qcv214/bsvcnn/pile/combined/fin_dat_ROI_',k))
  dat_allmat <- as.matrix(dat_allmat)
  
  num_part<- 2000
  num_test<- 2000
  set.seed(4)
  ind.to.use <- get_ind(num_part,num_test)
  
  print("stage 2")
  
  #subset data
  
  
  #Get bases matrices
  
  # deg_vec<- c(10,20,30,40)
  # a_vec  <- c(0.001,0.1,1)
  # b_vec  <- c(0.1,1,5,10,20,40,60,80,100)
  deg_vec<- c(30,40)
  a_vec  <- c(0.001,0.1,0.5,1,2,3)
  b_vec  <- c(1,5,10,20,40,60,80,100,120,140)
  param_grid <- as.matrix(expand.grid(deg_vec,a_vec,b_vec))
  
  
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bsvcnn/pile/combined/fin_mask_ROI_',k))
  nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)
  #re-scale the coordinates
  nb.centred<- apply(nb$maskcoords,2,norm.func)
  #re-centre each region
  # nb.centred <- sweep(nb.norm,2,apply(nb.norm,2,median),"-")
  
  
  res.pos <- matrix(,ncol = 2,nrow=nrow(param_grid))
  
  for(i in 1:nrow(param_grid)){
    
    tryCatch({
      poly_degree = param_grid[i,1]
      a_concentration= param_grid[i,2]
      b_smoothness= param_grid[i,3]
      dimension = 3
      #get psi
      psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
      #get lambda
      lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
      #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
      sqrt.lambda.nb <- sqrt(lambda.nb)
      bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
      bases.nb <- as.matrix(bases.nb)
      print("before cbind")
      #Get design matrix
      z.nb <- cbind(1,t(bases.nb%*%t(dat_allmat)))
      print("after cbind")
      train_Y <- dat.age[ind.to.use$train]
      train_Z <- z.nb[ind.to.use$train,]
      test_Y <- dat.age[ind.to.use$test]
      test_img <- dat_allmat[ind.to.use$test,]
      #get beta(v)
      hs_fit_SOI <- fast_horseshoe_lm(train_Y,train_Z)
      beta_fit <- data.frame(HS = crossprod(bases.nb,hs_fit_SOI$post_mean$betacoef[-1]))
      #in-sample prediction
      hs_in.pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + dat_allmat[ind.to.use$train,] %*%beta_fit$HS
      #out-sample prediction
      hs_pred_SOI <- hs_fit_SOI$post_mean$betacoef[1] + test_img %*%beta_fit$HS
      #Get correlation sqr
      pred_R2 <- c(Horseshoe = cor(hs_pred_SOI,test_Y)^2)
      out.hs <- c(unlist(t(as.matrix(rsqcal2(hs_in.pred_SOI,hs_pred_SOI,ind.to.use$train,ind.to.use$test)))),
                  as.numeric(sub('.*:', '', summary(beta_fit$HS))),
                  sum(abs(beta_fit$HS)>1e-5),
                  pred_R2[1])
      
      res.pos[i,1] <- out.hs[3]
      res.pos[i,2] <- out.hs[6]
    }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    
  }
  opt.pos <- c(which.max(res.pos[,1]),which.min(res.pos[,2]))
  # write.csv(opt.pos,paste0("/well/nichols/users/qcv214/bnn2/pile/gs23feb_ROI_",k,".csv"), row.names = FALSE)
  write.csv(opt.pos,paste0("/well/nichols/users/qcv214/bnn2/pile/gs25feb_ROI_",k,".csv"), row.names = FALSE)
  
}
  