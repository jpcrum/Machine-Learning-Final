###author: Liuqing Yang
tt=read.csv('data.csv',header=T)
tt=tt[,-33]
length(which(is.na(tt)=='T'))#check if there is missing value
#> length(which(is.na(tt)=='T'))
#[1] 0  
#there is no missing value

r1=length(which(tt$diagnosis=='M'))/length(tt$diagnosis)#ratio of malignant
r2=1-r1 #ratio of benign

#normalize data
nmlz=function(x){
  return((x-min(x))/(max(x)-min(x)))
}

tt_nmlz=apply(tt[,-c(1,2)],2,nmlz)
tt_nmlz=cbind.data.frame(tt[,c(1,2)],tt_nmlz)

write.csv(tt_nmlz,file = "data_nmlz.csv",row.names = F)#extract normalized data out.

#sepereate data for training and testing
n=length(tt_nmlz[,1])
n_train=floor(n*0.8) #let 80% in the original data for training.
n_test=n-n_train
set.seed(1234)
indx=sample(seq(1,n),n_train)
train=tt[indx,]
test=tt[-indx,]
train_nmlz=tt_nmlz[indx,]
test_nmlz=tt_nmlz[-indx,]
write.csv(train,file="train.csv",row.names=F)
write.csv(test,file="test.csv",row.names = F)
write.csv(train_nmlz,file="train_nmlz.csv",row.names=F)
write.csv(test_nmlz,file="test_nmlz.csv",row.names = F)



