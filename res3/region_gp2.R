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

mask.com<-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.reg <- sort(setdiff(unique(c(mask.com)),c(0)))
#load preset
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
part_use<-part_list[part_list$exist_vbm==1,] #4263 participants left
# 
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))


#load age
age_tab<-read_feather('/well/nichols/users/qcv214/bnn2/res3/age.feather')
dat.age <- age_tab$age

#data
dat_allmat <- as.matrix(read_feather('/well/nichols/users/qcv214/bnn2/res3/dat_rearranged.feather'))

print("stage 2")

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

dimension = 3
deg_vec<- c(10)
a_vec  <- c(0.001,0.01,0.1,0.5,1,2,3)
b_vec  <- 40
param_grid <- as.matrix(expand.grid(deg_vec,a_vec,b_vec))
gs.opt <- matrix(,ncol=2,nrow = length(mask.reg))
for(i in sort(mask.reg)){
  gs.opt[which(i==(mask.reg)),]<- as.matrix(read.csv(paste0("/well/nichols/users/qcv214/bnn2/res3/pile/gs5april_ROI_",i,".csv")))
}

bases.nb.temp <- matrix(,ncol=1,nrow= choose(max(deg_vec)+dimension,dimension)) 

for(i in mask.reg){
  poly_degree = param_grid[gs.opt[which(i==(mask.reg)),1],1] #######I forgot that it must the same poly degree....
  a_concentration= param_grid[gs.opt[which(i==(mask.reg)),1],2]
  b_smoothness= param_grid[gs.opt[which(i==(mask.reg)),1],3]
  mask.temp<-oro.nifti::readNIfTI(paste0('/well/nichols/users/qcv214/bnn2/res3/roi/mask_ROI_',i))
  
  nb <- find_brain_image_neighbors(img1, mask.temp, radius=1)
  #re-scale the coordinates
  nb.centred<- apply(nb$maskcoords,2,norm.func)
  #re-centre each region
  # nb.centred <- sweep(nb.norm,2,apply(nb.norm,2,median),"-") #actually maybe i don't need to centre it if it's bettern -1 and 1 already
  #get psi
  psi.mat.nb <- GP.eigen.funcs.fast(nb.centred, poly_degree = poly_degree, a = a_concentration, b = b_smoothness)
  #get lambda
  lambda.nb <- GP.eigen.value(poly_degree = poly_degree, a = a_concentration, b = b_smoothness, d = dimension)
  #Use Karhunen-Loeve expansion/Mercer's theorem to represent our GP as a combination of gaussian realisation, lambda and psi
  sqrt.lambda.nb <- sqrt(lambda.nb)
  bases.nb <- t(psi.mat.nb)*sqrt.lambda.nb
  if(nrow(bases.nb) < choose(max(deg_vec)+dimension,dimension)){
    empty.expansion <- matrix(0, nrow= choose(max(deg_vec)+dimension,dimension)-nrow(bases.nb), ncol= ncol(bases.nb))
    bases.nb <- rbind(bases.nb,empty.expansion)
  }
  #print(paste0("here, ",i))
  bases.nb.temp <- cbind(as.matrix(bases.nb.temp),as.matrix(bases.nb))
}
colnames(bases.nb.temp) <- NULL
bases.nb <- as.data.frame(bases.nb.temp[,-1])
bases.nb <- as.matrix(bases.nb)
print("before cbind")
#Get design matrix
z.nb <- cbind(1,t(bases.nb%*%t(dat_allmat)))
print("after cbind")

#subset data
num_part<- 2000
num_test<- 2000
set.seed(16)
ind.to.use <- get_ind(num_part,num_test)

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
write.csv(out.hs,paste0("/well/nichols/users/qcv214/bnn2/res3/pile/region_gp_11apr_seed16_",JobId,".csv"), row.names = FALSE)
write_feather(as.data.frame(beta_fit$HS),paste0("/well/nichols/users/qcv214/bnn2/res3/pile/region_gp_11apr_coef_seed16_",JobId,'.feather'))
