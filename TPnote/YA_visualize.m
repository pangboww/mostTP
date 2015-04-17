clear all;
clc;
load Yale_Faces.mat;

for c=1:15,
    ind_image = find(Y == c);
    
    sample = randi(length(ind_image));
    
    one_guy = reshape(X(ind_image(sample), :), 64, 64);
    
    subplot(2,8, c);
    imagesc(one_guy);colormap('Gray');
    axis image;
    set(gca,'xtick',[],'ytick',[]);
    title(['guy ' num2str(c)]);
end
