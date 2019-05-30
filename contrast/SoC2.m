clear all
clc
close all

% ������������
%load Yale_32
%load ORL_32       %400*1024(40��*10)
%load data/mat/AR_32       %3120*1024(120��*26)
%load AR_50
%load AR_50_zhe_12
%load AR_50_quan_12       %600*1024(50��*12)
%load COIL_20     %7200*1024(100��*72)
%load bio         %550*1024(22��*25)
%load UMIST     %575*1024(20��*)
%load AR_100
%load AR_100_wu_14
%load AR_100_zhe_12
%load AR_100_quan_12
load FERET

n=length(gnd)/max(gnd);%ÿһ���ж�������
class_n=max(gnd);
xx=10;%ʵ�����
accuracy = [];
big_accuracy = [];
small_accuracy = [];
big_times = [];
small_times = [];
for train_n=4
    %train_n=5;
    ave_accuracy=[];
    ave_times=[];
    mean_ave_accuracy=0;
    mean_ave_times=0;
    test_n=n-train_n;
    for ii=1:xx
        times=0;
        test_accuracy=0;
        t1=clock;
        
        
        
%         train=[];
%         test=[];
%         %[test, train] = crossvalind('holdOut',gnd,train_n/n);%��N���۲����������ѡȡ��������ڣ�P*N��������Ϊ���Լ�����PӦΪ0-1��ȱʡֵΪ0.5
%         for i=1:class_n
%             train=[train;(1:train_n)'+n*(i-1)];
%         end
%         for i=1:class_n
%             test=[test;(train_n+1:n)'+n*(i-1)];
%         end
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
        
        
        %train_n=15ʱ��ͨ�����㣬���ѵ������ռ14
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);
        kappa = 0.001;
        %k_near=size(train_data_labels,1)/4;%k����ȡһ��Ԫ��
        k_near=120;
        
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
        W_HE=zeros(class_n,size(test_samples,1));
        SW_HE=W_HE;
        index_w_he=W_HE;
        
        for i = 1:size(test_samples,1)
            W(:,i)=Project*test_samples(i,:)';
            for j=1:class_n%����������ı�ʾϵ��֮���
                W_HE(j,i)=sum(W(find(train_data_labels==j),i));
            end
%             [SW(:,i),index_w(:,i)]=sort(W(:,i),'descend');%�н�������
%             index_class(:,i)=floor((index_w(:,i)-1)./train_n)+1;%�н������ǩ
            [SW_HE(:,i),index_w_he(:,i)]=sort(W_HE(:,i),'descend');%�н�������
            
            if index_w_he(1,i)==test_data_labels(i,1)
                test_accuracy = test_accuracy + 1;
            end
%             if index_w_he(1,i)==test_data_labels(i,1)
%                 if index_class(1,i)~=test_data_labels(i,1)
%                     %disp(['ѭ��',num2str(i)]);
%                 end
%             end
%             if index_class(1,i)~=test_data_labels(i,1)
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
    mean_ave_accuracy=mean(ave_accuracy);
    ave_accuracy=ave_accuracy-mean_ave_accuracy;
    big_accuracy=[big_accuracy;max(ave_accuracy)];
    small_accuracy=[small_accuracy;min(ave_accuracy)];
    mean_ave_times=mean(ave_times);
    ave_times=ave_times-mean_ave_times;
    big_times=[big_times;max(ave_times)];
    small_times=[small_times;min(ave_times)];
    accuracy=[accuracy,mean_ave_accuracy];
    disp(['��������ռ��Ϊ ',num2str(train_n),' SoCʶ��ʱ��ave_times=',num2str(mean_ave_times)]);
    disp(['��������ռ��Ϊ ',num2str(train_n),' all_accuracy=',num2str(mean_ave_accuracy)]);
end
disp(['SoC2_accuracy=',num2str(accuracy)]);
disp('SoC2 done');
figure;
train_n=1:1:10;
plot(train_n, accuracy, '-*');
hold on;
% xx=1:1:20;
% for i=1:10
%     figure;
%     yy=W(1:20,i);
%     bar(xx,yy)
% end
% 
% %ȫ���������ع�
% huanyuan_quan=train_samples'*W(:,1);
% ju_huanyuan_quan = reshape(huanyuan_quan,32,32);
% imshow(ju_huanyuan_quan,[]);
% %��ȷ�������ع�������ϵ������󣬵�������ϵ��������
% huanyuan1=train_samples(1:6,:)'*W(1:6,1);
% ju_huanyuan1 = reshape(huanyuan1,32,32);
% imshow(ju_huanyuan1,[]);
% %ӵ�����ϵ��������ϵ���Ͳ���
% huanyuan2=train_samples(73:78,:)'*W(73:78,1);
% ju_huanyuan2 = reshape(huanyuan2,32,32);
% imshow(ju_huanyuan2,[]);
% %ϵ�����Ƚ�С
% huanyuan3=train_samples(61:66,:)'*W(61:66,1);
% ju_huanyuan3 = reshape(huanyuan3,32,32);
% imshow(ju_huanyuan3,[]);