clear all
clc
close all

% 加载样本数据
%load ORL_32       %400*1024(40类*10)
%load AR_50       %3120*1024(120类*26)
%load AR_50_zhe_12
%load AR_50_quan_12       %600*1024(50类*12)
%load COIL_20     %7200*1024(100类*72)
%load bio         %550*1024(22类*25)
%load UMIST     %575*1024(20类*)
%load AR_100_wu_14
%load AR_100_zhe_12
%load AR_100_quan_12
load mnist_40
%load Yale_32
%load data/mat/PIE_32      %11554*1024(68类*170)
%load data/mat/Yale_32     %165*1024(15类*11)
%load data/mat/YaleB_32    %2414*1024(38类*64)
% fea2=[];
% gnd2=[];
% index2=floor(rand(160,1)*160)+1;
% for i=1:57
%     index1=find(gnd==i);
%     %index2=floor(randperm(160,1)*160)+1;
%     %index1(index2(1:46,1),1)
%     %size(fea(index1(index2(1:46,1),1),:))
%     %index2=[];
%     fea2=[fea2;fea(index1(index2(1:46,1)),:)];
%     %fea2((i-1)*46+1:i*46,:)=fea(index1(index2(1:46,1)));
%     gnd2=[gnd2;ones(46,1)*i];
% end
% fea=[];
% fea=fea2;
% gnd=[];
% gnd=gnd2;
n=length(gnd)/max(gnd);%每一类有多少样本
xx=10;%实验次数
accuracy = [];
big_accuracy = [];
small_accuracy = [];
big_times = [];
small_times = [];
for train_n=10%训练样本数占比
    ave_accuracy=[];
    ave_times=[];
    mean_ave_accuracy=0;
    mean_ave_times=0;
    test_n=n-train_n;
    for ii=1:xx
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
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);
        %test_samples=add_noise_gaussian(test_samples,0.03);
        kappa = 0.001;
%         % PCA降维
%         options = [];
%         %options.PCARatio = 0.9;
%         options.ReducedDim = dim;
%         [eigvector_PCA, ~, ~, new_train_samples] = PCA(train_samples, options);
%         % 计算特征向量，即变换矩阵
%         train_samples = train_samples * eigvector_PCA;
%         test_samples = test_samples * eigvector_PCA;

            %train_samples = train_samples./( repmat(sqrt(sum(train_samples.*train_samples,2)), [1,size(train_samples,2)]) );
        %train_samples = train_samples./255;
        %sum(train_samples,2)
            %test_samples = test_samples./( repmat(sqrt(sum(test_samples.*test_samples,2)), [1,size(test_samples,2)]) );
        %test_samples = test_samples./255;
        %计算投影矩阵,前半部分是样本数方阵求逆，表示用所有样本来表示，不同于协方差是样本维数的方阵
        Project = inv(train_samples*train_samples'+kappa*eye(size(train_samples,1)))*train_samples;
        %Project = Project(1:dim,:);
        % 识别过程
        
        accuraty = 0;
        for i = 1:size(test_samples,1)
            y = test_samples(i,:)';
            %x=train_samples(i,:)';
            coef = Project*y;
            for ci = 1:max(train_data_labels)
                coef_c = coef(train_data_labels==ci);
                Dc = train_samples(train_data_labels==ci,:)';
                %error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
                error(ci) = norm(y-Dc*coef_c,2);
            end
            index      =  find(error==min(error));
            id         =  index(1);
            if id==test_data_labels(i)
                accuraty=accuraty+1;
            end
        end
        test_accuracy=accuraty/size(test_samples,1);
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
    disp(['测试样本占比为 ',num2str(train_n),' CRC识别时间ave_times=',num2str(mean_ave_times)]);
    disp(['测试样本占比为 ',num2str(train_n),' all_accuracy=',num2str(mean_ave_accuracy)]);
end
disp(['CRC_accuracy=',num2str(accuracy)]);
disp('CRCdone');
dim = 10:10:100;
plot(dim, accuracy, '-*');
hold on;



