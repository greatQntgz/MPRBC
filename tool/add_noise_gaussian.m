function [noise_fea_gaussian] = add_noise_gaussian(fea, salt)
% 加载样本数据
%load data/mat/ORL_32      %400*1024(40类*10)
%load data/mat/AR_32       %3120*1024(120类*26)
%load data/mat/COIL100     %7200*1024(100类*72)
%load data/mat/PIE_32      %11554*1024(68类*170)
%load data/mat/Yale_32     %165*1024(15类*11)
%load data/mat/YaleB_32    %2414*1024(38类*64)
noise_fea_gaussian=zeros(size(fea));
for i=1:size(fea,1)
    F=fea(i,:);
    F=reshape(F,32,32);
    F=uint8(F);
%     subplot(121);
%     imshow(F,[]);
%     title('原图'); 
    F = imnoise(F,'gaussian',0,salt); %给图像加入噪声 0.1为可以修改的参数 
%     subplot(122);
%     imshow(F,[]);
%     title('加噪声之后');
    F=double(F);
    F=reshape(F,1,1024);
    noise_fea_gaussian(i,:)=F;
end
%noise_fea_salt