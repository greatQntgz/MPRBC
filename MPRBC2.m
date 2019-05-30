clear all
clc
close all
%在TPLRMC的基础上，改成多段类和最大分类，每次样本数除以参数chu，跳出条件c<=break_n
addpath(genpath(pwd));%将当前文件夹下的所有文件夹都包括进函数的目录
% 加载样本数据
%load Yale_32   %165*1024(15类*11)
%load UMIST     %575*1024(20类*)
%load bio         %550*1024(22类*25)
load ORL_32       %400*1024(40类*10)
%load AR_100     %2600*1024(100类*26)
%load FERET     %1400*1024(200类*7)



%给随机的50%图片（包括训练和测试）加噪声
% index1=floor(rand(size(gnd,1)-1,1)*(size(gnd,1)-1))+1;
% index2=index1(1:floor(size(gnd,1)/2));
% fea2=add_noise_salt(fea(index2,:),0.1);
% fea(index2,:)=fea2;

n=length(gnd)/max(gnd);%每一类有多少样本
class_n=max(gnd);%类别数
xx=10;%实验次数
accuracy = [];
time=[];
alpha=1;
train_n=5;%每类训练样本数
for break_n=4:4:40%设定算法结束的类别数阈值
ave_accuracy=[];
ave_times=[]; 
test_n=n-train_n;
    for ii=1:xx
        test_accuracy=0;
        t1=clock;

        %划分训练集和测试集
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
        %也可以用该方法，在个别数据集上，由于除数的原因会出错
        %[test, train] = crossvalind('holdOut',gnd,train_n/n);%从N个观察样本中随机选取（或近似于）P*N个样本作为测试集。故P应为0-1，缺省值为0.5
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);

        %加噪声函数
        %test_samples=add_noise_gaussian(test_samples,0.03);

        kappa = 0.001;
        %初始的投影矩阵，放到循环外面，不需要重复计算
        Project_yuan = (train_samples*train_samples'+kappa*eye(size(train_samples,1)))\train_samples;

        for i = 1:size(test_samples,1)
            c=class_n;%计算过程中的类别数，初始为总类别数
            X_K=train_samples;%迭代过程中的训练样本，初始为总训练样本
            index=(1:1:class_n)';
            big_k_n=0;
            cc=1000;
            Project=Project_yuan;
            while c>=break_n
                if cc==c%如果类别数没有变化则迭代结束，为了防止出现死循环
                    break;
                end
                cc=c;
                Wi=zeros(size(X_K,1),1);%迭代过程中的投影矩阵
                sum_W=zeros(c,1);
                s_sum_W=sum_W;
                index_s_sum_W=sum_W;


                Wi(:,1)=Project*test_samples(i,:)';
                for k=1:c
                    sum_W(k,1)=sum(Wi((1+(k-1)*train_n):k*train_n,1));%每个类的系数求和
                end
                [s_sum_W(:,1),index_s_sum_W(:,1)]=sort(sum_W(:,1),'descend');%类系数排序
                ge=sum(s_sum_W(:,1)>0);%类系数大于零的总个数
                if alpha==1%如果alpha==1，则取类系数为正的所有类
                    c=ge;
                else
                    zheng=sum(s_sum_W(1:ge,1));
                    he=0;
                    for a=1:c
                        he=he+s_sum_W(a,1);
                        if(he>=alpha*zheng);
                            break;
                        end
                    end
                    c=a-1;
                end
                k_n=zeros(c,1);
                hang=zeros(c*train_n,1);
                k_n(:,1)=index(index_s_sum_W(1:c,1),1);
                t=1;
                for nn=1:size(k_n,1)
                    for n_train=1:train_n
                        hang(t,1)=k_n(nn,1)*train_n-train_n+n_train;
                        t=t+1;
                    end
                end
                X_K=train_samples(hang,:);%新的训练样本
                index=k_n;
                big_k_n=k_n(1,1);%拥有最大类系数的类，就是预测标签
                if c>=break_n
                    Project = (X_K*X_K'+kappa*eye(size(X_K,1)))\X_K;
                end
            end
            if test_data_labels(i,1)==big_k_n
                test_accuracy = test_accuracy + 1;
            end
        end
        test_accuracy = test_accuracy/(test_n*class_n);%一次循环下来的平均识别精度
        disp(['循环',num2str(ii),' test_accuracy=',num2str(test_accuracy)]);
        ave_accuracy=[ave_accuracy;test_accuracy]; 
        t2=clock;
        times=etime(t2,t1);
        ave_times=[ave_times;times];
    end
mean_ave_accuracy=mean(ave_accuracy);
mean_ave_times=mean(ave_times);
accuracy=[accuracy,mean_ave_accuracy];
time=[time,mean_ave_times];
disp(['测试样本占比为 ',num2str(train_n),'break=',num2str(break_n),' MPRBC识别时间=',num2str(mean_ave_times)]);
disp(['测试样本占比为 ',num2str(train_n),'break=',num2str(break_n),' MPRBC识别率=',num2str(mean_ave_accuracy)]);
end
disp(['MPRBC_accuracy=',num2str(accuracy)]);
disp('MPRBC done');
figure;
break_n=4:4:40;
plot(break_n, accuracy, '-*');
hold on;
