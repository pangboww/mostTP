%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                         Euclidienne                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

load('Yale_Faces.mat');

[x_av, y_av, x_t, y_t] = split_data(X, Y, 0.8);

K = 1:1:10;
n_K = length(K);
B = 5;

err_a = zeros(n_K, B);
err_v = zeros(n_K, B);


for b = 1:B
    [x_a, y_a, x_v, y_v, indices] = split_data_fold_CV(x_av, y_av, 5, b);
    for k = 1:n_K
        mdl = fitcknn(x_a,y_a,'NumNeighbors',K(k),'Distance','euclidean');
        y_ap = predict(mdl,x_a);
        y_vp = predict(mdl,x_v);
        
        conf_matrix_train = confusionmat(y_a,y_ap) ./ size(x_a, 1);
        conf_matrix_valid = confusionmat(y_v,y_vp) ./ size(x_v, 1);
        
        err_a(k,b) = 1 - sum(diag(conf_matrix_train));
        err_v(k,b) = 1 - sum(diag(conf_matrix_valid));
    end
end

plot(K,mean(err_a,2),'r');hold on;
plot(K,mean(err_v,2),'b');
legend('training error','validation error','Location','southeast');

[val_mmin, ind_min] = min(mean(err_v, 2));
mdl = fitcknn(x_av,y_av,'NumNeighbors',K(ind_min),'Distance','euclidean');
y_tp = predict(mdl,x_t);
conf_matrix_test = confusionmat(y_t,y_tp) ./ size(x_t,1);
err_t = 1 - sum(diag(conf_matrix_test));






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            Cityblock                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

load('Yale_Faces.mat');

[x_av, y_av, x_t, y_t] = split_data(X, Y, 0.8);

K = 1:10;
n_K = length(K);
B = 5;

err_a = zeros(n_K, B);
err_v = zeros(n_K, B);


for b = 1:B
    [x_a, y_a, x_v, y_v, indices] = split_data_fold_CV(x_av, y_av, 5, b);
    for k = 1:n_K
        mdl = fitcknn(x_a,y_a,'NumNeighbors',K(k),'Distance','cityblock');
        y_ap = predict(mdl,x_a);
        y_vp = predict(mdl,x_v);
        
        conf_matrix_train = confusionmat(y_a,y_ap) ./ size(x_a, 1);
        conf_matrix_valid = confusionmat(y_v,y_vp) ./ size(x_v, 1);
        
        err_a(k,b) = 1 - sum(diag(conf_matrix_train));
        err_v(k,b) = 1 - sum(diag(conf_matrix_valid));
    end
end


plot(K,mean(err_a,2),'r');hold on;
plot(K,mean(err_v,2),'b');
legend('training error','validation error','Location','southeast');

[val_mmin, ind_min] = min(mean(err_v, 2));
mdl = fitcknn(x_av,y_av,'NumNeighbors',K(ind_min),'Distance','cityblock');
y_tp = predict(mdl,x_t);
conf_matrix_test = confusionmat(y_t,y_tp) ./ size(x_t,1);
err_t = 1 - sum(diag(conf_matrix_test));






