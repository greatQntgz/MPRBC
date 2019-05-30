function show(data)
for i=[1:26]
    F=data(i,:);
    F=reshape(F,32,32);
    F=uint8(F);
    figure;
    imshow(F,[]);
end