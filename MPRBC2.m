clear all
clc
close all
%��TPLRMC�Ļ����ϣ��ĳɶ����������࣬ÿ�����������Բ���chu����������c<=break_n
addpath(genpath(pwd));%����ǰ�ļ����µ������ļ��ж�������������Ŀ¼
% ������������
%load Yale_32   %165*1024(15��*11)
%load UMIST     %575*1024(20��*)
%load bio         %550*1024(22��*25)
load ORL_32       %400*1024(40��*10)
%load AR_100     %2600*1024(100��*26)
%load FERET     %1400*1024(200��*7)



%�������50%ͼƬ������ѵ���Ͳ��ԣ�������
% index1=floor(rand(size(gnd,1)-1,1)*(size(gnd,1)-1))+1;
% index2=index1(1:floor(size(gnd,1)/2));
% fea2=add_noise_salt(fea(index2,:),0.1);
% fea(index2,:)=fea2;

n=length(gnd)/max(gnd);%ÿһ���ж�������
class_n=max(gnd);%�����
xx=10;%ʵ�����
accuracy = [];
time=[];
alpha=1;
train_n=5;%ÿ��ѵ��������
for break_n=4:4:40%�趨�㷨�������������ֵ
ave_accuracy=[];
ave_times=[]; 
test_n=n-train_n;
    for ii=1:xx
        test_accuracy=0;
        t1=clock;

        %����ѵ�����Ͳ��Լ�
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
        %Ҳ�����ø÷������ڸ������ݼ��ϣ����ڳ�����ԭ������
        %[test, train] = crossvalind('holdOut',gnd,train_n/n);%��N���۲����������ѡȡ��������ڣ�P*N��������Ϊ���Լ�����PӦΪ0-1��ȱʡֵΪ0.5
        train_samples=fea(train,:);
        train_data_labels=gnd(train);
        test_samples=fea(test,:);
        test_data_labels=gnd(test);
        [nSmp, nFea] = size(train_samples);

        %����������
        %test_samples=add_noise_gaussian(test_samples,0.03);

        kappa = 0.001;
        %��ʼ��ͶӰ���󣬷ŵ�ѭ�����棬����Ҫ�ظ�����
        Project_yuan = (train_samples*train_samples'+kappa*eye(size(train_samples,1)))\train_samples;

        for i = 1:size(test_samples,1)
            c=class_n;%��������е����������ʼΪ�������
            X_K=train_samples;%���������е�ѵ����������ʼΪ��ѵ������
            index=(1:1:class_n)';
            big_k_n=0;
            cc=1000;
            Project=Project_yuan;
            while c>=break_n
                if cc==c%��������û�б仯�����������Ϊ�˷�ֹ������ѭ��
                    break;
                end
                cc=c;
                Wi=zeros(size(X_K,1),1);%���������е�ͶӰ����
                sum_W=zeros(c,1);
                s_sum_W=sum_W;
                index_s_sum_W=sum_W;


                Wi(:,1)=Project*test_samples(i,:)';
                for k=1:c
                    sum_W(k,1)=sum(Wi((1+(k-1)*train_n):k*train_n,1));%ÿ�����ϵ�����
                end
                [s_sum_W(:,1),index_s_sum_W(:,1)]=sort(sum_W(:,1),'descend');%��ϵ������
                ge=sum(s_sum_W(:,1)>0);%��ϵ����������ܸ���
                if alpha==1%���alpha==1����ȡ��ϵ��Ϊ����������
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
                X_K=train_samples(hang,:);%�µ�ѵ������
                index=k_n;
                big_k_n=k_n(1,1);%ӵ�������ϵ�����࣬����Ԥ���ǩ
                if c>=break_n
                    Project = (X_K*X_K'+kappa*eye(size(X_K,1)))\X_K;
                end
            end
            if test_data_labels(i,1)==big_k_n
                test_accuracy = test_accuracy + 1;
            end
        end
        test_accuracy = test_accuracy/(test_n*class_n);%һ��ѭ��������ƽ��ʶ�𾫶�
        disp(['ѭ��',num2str(ii),' test_accuracy=',num2str(test_accuracy)]);
        ave_accuracy=[ave_accuracy;test_accuracy]; 
        t2=clock;
        times=etime(t2,t1);
        ave_times=[ave_times;times];
    end
mean_ave_accuracy=mean(ave_accuracy);
mean_ave_times=mean(ave_times);
accuracy=[accuracy,mean_ave_accuracy];
time=[time,mean_ave_times];
disp(['��������ռ��Ϊ ',num2str(train_n),'break=',num2str(break_n),' MPRBCʶ��ʱ��=',num2str(mean_ave_times)]);
disp(['��������ռ��Ϊ ',num2str(train_n),'break=',num2str(break_n),' MPRBCʶ����=',num2str(mean_ave_accuracy)]);
end
disp(['MPRBC_accuracy=',num2str(accuracy)]);
disp('MPRBC done');
figure;
break_n=4:4:40;
plot(break_n, accuracy, '-*');
hold on;
