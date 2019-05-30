function [noise_fea_gaussian] = add_noise_gaussian(fea, salt)
% ������������
%load data/mat/ORL_32      %400*1024(40��*10)
%load data/mat/AR_32       %3120*1024(120��*26)
%load data/mat/COIL100     %7200*1024(100��*72)
%load data/mat/PIE_32      %11554*1024(68��*170)
%load data/mat/Yale_32     %165*1024(15��*11)
%load data/mat/YaleB_32    %2414*1024(38��*64)
noise_fea_gaussian=zeros(size(fea));
for i=1:size(fea,1)
    F=fea(i,:);
    F=reshape(F,32,32);
    F=uint8(F);
%     subplot(121);
%     imshow(F,[]);
%     title('ԭͼ'); 
    F = imnoise(F,'gaussian',0,salt); %��ͼ��������� 0.1Ϊ�����޸ĵĲ��� 
%     subplot(122);
%     imshow(F,[]);
%     title('������֮��');
    F=double(F);
    F=reshape(F,1,1024);
    noise_fea_gaussian(i,:)=F;
end
%noise_fea_salt