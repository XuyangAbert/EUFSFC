close all;
clear all;
clc;
tic
% Raw data set Input
sample=dlmread('Prostate-GE.csv');

[N1,L1]=size(sample);
D1=L1-1;
Raw_data=sample(:,1:D1);


% Reduced-dimensional data set Input
Reduce_sample=dlmread('Extract_Pro.txt');
[N2,L2]=size(Reduce_sample);
D2=L2-1;
Reduce_data=Reduce_sample(:,1:D2);

% Target Class Labels
Y=sample(:,L1);
% Split it into training and testing data set
for it = 1:20
indices = crossvalind('Kfold',Y,10);
cp1 = classperf(Y);
cp2 = classperf(Y);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    T1=fitctree(Raw_data(train,:),Y(train));
    T2=fitctree(Reduce_data(train,:), Y(train));
    
    class1 = predict(T1, Raw_data(test,:));
    
    class2 = predict(T2, Reduce_data(test,:));
    classperf(cp1,class1,test);
    classperf(cp2,class2,test);
end
Error1(it) = cp1.ErrorRate;
Error2(it) = cp2.ErrorRate;

end
E1 = mean(Error1);
E2 = mean(Error2);

Acc1 = 1-E1;
Acc2 = 1-E2;
% csvwrite('F-Score-DT-90.csv',[Acc1 Acc2]);
toc