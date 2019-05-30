clear all
clc
close all

% ������������
%load ORL_32       %400*1024(40��*10)
%load bio
%load UMIST
load FERET
%load Yale_32
%load PIE_42
%load PIE_32
%load AR_100
%load COIL_20
%load AR_100_wu_14
%load AR_100_zhe_12
%load AR_100_quan_12


%load data/mat/AR_32       %3120*1024(120��*26)
%load data/mat/AR_100
%load data/mat/COIL100     %7200*1024(100��*72)
%load data/mat/PIE_32      %11554*1024(68��*170)
%load data/mat/Yale_32     %165*1024(15��*11)
%load data/mat/YaleB_32    %2414*1024(38��*64)
%show(fea);
% fea2=fea(1:1300,:);
% gnd2=gnd(1:1300,:);
% fea=[];
% gnd=[];
% fea=fea2;
% gnd=gnd2;
% fea2=[];
% gnd2=[];

%�������50%ͼƬ������ѵ���Ͳ��ԣ�������
% index1=floor(rand(size(gnd,1)-1,1)*(size(gnd,1)-1))+1;
% index2=index1(1:floor(size(gnd,1)/2));
% fea2=add_noise_salt(fea(index2,:),0.1);
% fea(index2,:)=fea2;



n=length(gnd)/max(gnd);%ÿһ���ж�������
class_n=max(gnd);
xx=10;%ʵ�����
accuracy = [];
time=[];
big_accuracy = [];
small_accuracy = [];
big_times = [];
small_times = [];
xs = [];
xs_count = [];
for train_n=4
    %train_n=5;
    for k_near=20:20:200
    ave_accuracy=[];
    ave_times=[];
    mean_ave_accuracy=0;
    mean_ave_times=0;
    test_n=n-train_n;
    xsi = zeros(1,(size(gnd,1)-class_n*train_n));
    for ii=1:xx
        times=0;
        test_accuracy=0;
        t1=clock;
%         train=[];
%         test=[];
        
        
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
        
        
        %[test, train] = crossvalind('holdOut',gnd,train_n/n);%��N���۲����������ѡȡ��������ڣ�P*N��������Ϊ���Լ�����PӦΪ0-1��ȱʡֵΪ0.5
%         for i=1:class_n
%             train=[train;(1:train_n)'+n*(i-1)];
%         end
%         for i=1:class_n
%             test=[test;(train_n+1:n)'+n*(i-1)];
%         end
        %train_n=15ʱ��ͨ�����㣬���ѵ������ռ14
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);
        %test_samples=add_noise_gaussian(test_samples,0.03);
        kappa = 0.001;
        %k_near=size(train_data_labels,1)/4;%k����ȡһ��Ԫ��
        %k_near=24;
        Project = (train_samples*train_samples'+kappa*eye(size(train_samples,1)))\train_samples;

        
        % W=zeros(nSmp,size(test_samples,1));
        % for i = 1:size(test_samples,1)
        %     W(:,i)=Project*test_samples(i,:)';
        % end
        % [SW,index]=sort(W,'descend');%�н�������
        % SW_K=SW(1:k_near,:);
        % index_k=index(1:k_near,:);


        W=zeros(nSmp,size(test_samples,1));
        SW=W;
        index_w=W;
        index_class=W;
        W_K=W(1:k_near,:);
        SW_K=W_K;
        index_w_k=W_K;
        index_w_w=W_K;%�ж�����xi�Ĺ����ǲ���һ����ǰ��
        index_k_class=W_K;
        W_K_HE=zeros(class_n,size(test_samples,1));
        
        for i = 1:size(test_samples,1)
            W(:,i)=Project*test_samples(i,:)';
            [SW(:,i),index_w(:,i)]=sort(W(:,i),'descend');%�н�������
            index_class(:,i)=floor((index_w(:,i)-1)./train_n)+1;%�н������ǩ
            X_K=train_samples(index_w(1:k_near,i),:);%���״Ӵ�С���������ţ�����ÿһ����һ��
        %     SW_K(:,i)=SW(1:k_near,i);
        %     index_k(:,i)=index_w(1:k_near,i);
        %     X_K=train_samples(index_k(:,i),:);%���״Ӵ�С���������ţ�����ÿһ����һ��

            Project_k = (X_K*X_K'+kappa*eye(size(X_K,1)))\X_K;
            W_K(:,i)=Project_k*test_samples(i,:)';%k���ڵı�ʾϵ��
            [SW_K(:,i),index_w_k(:,i)]=sort(W_K(:,i),'descend');%�н�������

            index_w_w(:,i)=index_w(index_w_k(:,i),i);

            index_k_class(:,i)=floor((index_w(1:k_near,i)-1)./train_n)+1;%k���ڵ����ǩ
            xsi(1,i)=xsi(1,i)+length(unique(index_k_class(:,i)));
            for j=1:k_near
                W_K_HE(index_k_class(j,i),i)=W_K_HE(index_k_class(j,i),i)+W_K(j,i);
            end
            if test_data_labels(i,1)==find(W_K_HE(:,i)==max(W_K_HE(:,i)),1)
                test_accuracy = test_accuracy + 1;
            end
%             if test_data_labels(i,1)~=find(W_K_HE(:,i)==max(W_K_HE(:,i)),1)
%                 disp(['ѭ��',num2str(i)]);
%             end
%             if test_data_labels(i,1)~=find(W_K_HE(:,i)==max(W_K_HE(:,i)),1)
%                 disp(['i=',num2str(i),' aero=',num2str(find(W_K_HE(:,i)==max(W_K_HE(:,i)),1))]);
%             end
        end

        test_accuracy = test_accuracy/(test_n*class_n);
        disp(['ѭ��',num2str(ii),' test_accuracy=',num2str(test_accuracy)]);
        ave_accuracy=[ave_accuracy;test_accuracy]; 
        t2=clock;
        times=etime(t2,t1);
        ave_times=[ave_times;times];
    end
    
    xsi=xsi/xx;
    xs_count=[xs_count;sum(xsi)/size(test_samples,1)];
    %xsi=xsi/size(test_samples,1)/xx;
    xs=[xs;xsi];
    
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
    disp(['��������ռ��Ϊ ',num2str(train_n),'near=',num2str(k_near),' TPLRMC2ʶ��ʱ��ave_times=',num2str(mean_ave_times)]);
    disp(['��������ռ��Ϊ ',num2str(train_n),'near=',num2str(k_near),' all_accuracy=',num2str(mean_ave_accuracy)]);
    end
end
disp('TPLRMC done');
figure;
k_near=20:20:200;
plot(k_near, accuracy, '-*');
hold on;
% xx=1:1:20;
% for i=1:10
%     figure;
%     yy=W(1:20,i);
%     bar(xx,yy)
% end