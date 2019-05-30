clear all
clc
close all


% 加载样本数据
%load ORL_32       %400*1024(40类*10)%3:120/5:200
%load AR_50       %3120*1024(120类*26)%9:400/13:550
%load AR_50_zhe_12%4:120/6:120
%load AR_50_quan_12       %600*1024(50类*12)%4:150/6:250
%load COIL_20     %7200*1024(100类*72)
%load bio         %550*1024(22类*25)
load UMIST     %575*1024(20类*)
%load data/mat/PIE_32      %11554*1024(68类*170)
%load Yale_32     %165*1024(15类*11)
%load AR_100_wu_14
%load AR_100_zhe_12
%load AR_100_quan_12
%load data/mat/YaleB_32    %2414*1024(38类*64)
%show(fea);
n=length(gnd)/max(gnd);%每一类有多少样本
class_n=max(gnd);
xx=10;%实验次数
accuracy = [];
time=[];
big_accuracy = [];
small_accuracy = [];
big_times = [];
small_times = [];
for train_n=10
    %train_n=5;
    %for k_near=train_n/5*10:train_n/5*10:train_n/5*100
    for k_near=20:20:200
    ave_accuracy=[];
    ave_times=[];
    mean_ave_accuracy=0;
    mean_ave_times=0;
    test_n=n-train_n;
    
    for ii=1:xx
        times=0;
        test_accuracy=0;
        t1=clock;
        
        
        test=[];
        train=[];
        for cc=1:max(gnd)
            gnd1=gnd(gnd==cc);
            nn=length(gnd1);
            [test1,train1]=crossvalind('holdOut',gnd1,train_n/nn);
            train=[train;train1];
            test=[test;test1];
        end
        train=logical(train);
        test=logical(test);
        
        
        %[test, train] = crossvalind('holdOut',gnd,train_n/n);%从N个观察样本中随机选取（或近似于）P*N个样本作为测试集。故P应为0-1，缺省值为0.5
        %train_n=15时，通过运算，结果训练样本占14
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);
        test_samples=add_noise_gaussian(test_samples,0.03);
        kappa = 0.001;
        %k_near=size(train_data_labels,1)/4;%k近邻取一半元素
        %k_near=80;
        %k_s=2;
        %Project = (train_samples*train_samples'+kappa*eye(size(train_samples,1)))\train_samples;

        
        % W=zeros(nSmp,size(test_samples,1));
        % for i = 1:size(test_samples,1)
        %     W(:,i)=Project*test_samples(i,:)';
        % end
        % [SW,index]=sort(W,'descend');%列降序排列
        % SW_K=SW(1:k_near,:);
        % index_k=index(1:k_near,:);


        W=zeros(nSmp,size(test_samples,1));
        SW=W;
        %SW_S=W;
        index_w=W;
        W_CAN=W;
        SW_CAN=W_CAN;
        index_w_can=W_CAN;
        W_K=W(1:k_near,:);
        W_K_CAN=zeros(k_near,nFea);
        index_k_class=W_K;
        W_K_CAN_HE=zeros(class_n,nFea);
        W_K_CLASS_CAN=zeros(class_n,size(test_samples,1));
        SW_K_CLASS_CAN=W_K_CLASS_CAN;
        index_w_k_class_can=W_K_CLASS_CAN;
        tau=.35;
        tolA = 1.e-3;
        for i = 1:size(test_samples,1)
            y = test_samples(i,:)';
            [coef,coef_debias,obj_GPSR_Basic,times_GPSR_Basic,debias_s,mses_GPSR_Basic]= ...
                GPSR_Basic(y,train_samples',tau,...
                    'Debias',0,...
                    'StopCriterion',1,...
                    'ToleranceA',tolA);
            %class_s=[];
            %count=1;
            X_K=[];
            %index_w_w=zeros(k_near,size(test_samples,1));%判断两步xi的贡献是不是一样在前面
            %W(:,i)=Project*test_samples(i,:)';
            W(:,i)=coef;
            for k_can=1:nSmp
                W_CAN(k_can,i) = norm(test_samples(i,:)-W(k_can,i)*train_samples(k_can,:),2);
            end
            [SW_CAN(:,i),index_w_can(:,i)]=sort(W_CAN(:,i));%列升序排列
            %[SW_S(:,i),index_w_s(:,i)]=sort(W(:,i));%列升序排列
            %index_class_b(:,i)=floor((index_w_b(:,i)-1)./train_n)+1;%列降序类标签
            %index_class_s(:,i)=floor((index_w_s(:,i)-1)./train_n)+1;%列升序类标签
%             for k_near_n=1:k_near
%                 W_CAN(k_near_n,i) = norm(test_samples(i,:)-SW(k_near_n,i)*train_samples(index_w(k_near_n,i),:),2);
%             end
%             [SW_CAN(:,i),index_w_can(:,i)]=sort(W_CAN(:,i));%残差按列升序排
%             index(:,i)=index_w(index_w_can(:,i),i);
%             index_k_class(:,i)=floor((index(:,i)-1)./train_n)+1;
            X_K=train_samples(index_w_can(1:k_near,i),:);
            %Project_k = (X_K*X_K'+kappa*eye(size(X_K,1)))\X_K;
            %W_K(:,i)=Project_k*test_samples(i,:)';
            [coef2,coef_debias,obj_GPSR_Basic,times_GPSR_Basic,debias_s,mses_GPSR_Basic]= ...
                GPSR_Basic(y,X_K',tau,...
                    'Debias',0,...
                    'StopCriterion',1,...
                    'ToleranceA',tolA);
            W_K(:,i)=coef2;
%             for k_near_n=1:k_near %K近邻的样本乘以系数
%                 W_K_CAN(k_near_n,:) = W_K(k_near_n,i)*X_K(k_near_n,:);
%             end
            index_k_class(:,i)=floor((index_w_can(1:k_near,i)-1)./train_n)+1;%k近邻的类标签
%             W_K_CAN_HE=zeros(class_n,nFea);
%             for j=1:k_near %类样本和系数乘积之和
%                 W_K_CAN_HE(index_k_class(j,i),:)=W_K_CAN_HE(index_k_class(j,i),:)+W_K_CAN(j,:);
%             end
%             
%             
%             for k_class_n=1:class_n %类残差
%                 if W_K_CAN_HE(k_class_n,i)==0
%                     W_K_CLASS_CAN(k_class_n,i)=realmax;
%                 else
%                     W_K_CLASS_CAN(k_class_n,i) = norm(test_samples(i,:)-W_K_CAN_HE(k_class_n,i),2);
%                 end
%             end
%             
%             [SW_K_CLASS_CAN(:,i),index_w_k_class_can(:,i)]=sort(W_K_CLASS_CAN(:,i));%列升序排列
%             
            error=[];
            coef_c=[];
            Dc=[];
            for ci = 1:max(index_k_class(:,i))
                coef_c = W_K(index_k_class(:,i)==ci,i);
                Dc = X_K(index_k_class(:,i)==ci,:)';
                %error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
                error(ci) = norm(test_samples(i,:)'-Dc*coef_c,2);
            end
            index      =  find(error==min(error));
            id         =  index(1);

%             if test_data_labels(i,1)==index_w_k_class_can(1,i)
            if test_data_labels(i,1)==id
                test_accuracy = test_accuracy + 1;
            end
%             [SW_CAN(:,i),index_w_can(:,i)]=sort(W_CAN(:,i));%残差按列升序排
%             index(:,i)=index_w(index_w_can(:,i),i);
%             index_k_class(:,i)=floor((index(:,i)-1)./train_n)+1;
            
            
            
%             if test_data_labels(i,1)==index_k_class(1,i)
%                 test_accuracy = test_accuracy + 1;
%             end
        end
        test_accuracy = test_accuracy/(test_n*class_n);
        disp(['循环',num2str(ii),' test_accuracy=',num2str(test_accuracy)]);
        ave_accuracy=[ave_accuracy;test_accuracy]; 
        t2=clock;
        times=etime(t2,t1);
        ave_times=[ave_times;times];
    end
    
    mean_ave_accuracy=mean(ave_accuracy);
    ave_accuracy=ave_accuracy-mean_ave_accuracy;
    big_accuracy=[big_accuracy;max(ave_accuracy)];
    small_accuracy=[small_accuracy;min(ave_accuracy)];
    mean_ave_times=mean(ave_times);
    ave_times=ave_times-mean_ave_times;
    big_times=[big_times;max(ave_times)];
    small_times=[small_times;min(ave_times)];
    accuracy=[accuracy,mean_ave_accuracy];
    time=[time,mean_ave_times];
    disp(['测试样本占比为 ',num2str(train_n),'near=',num2str(k_near),' TPTSR识别时间ave_times=',num2str(mean_ave_times)]);
    disp(['测试样本占比为 ',num2str(train_n),'near=',num2str(k_near),' all_accuracy=',num2str(mean_ave_accuracy)]);
    end
end
disp('TPLRMC done');
figure;
k_near=20:20:200;
plot(k_near, accuracy, '-*');
hold on;
% accuracy2=accuracy;