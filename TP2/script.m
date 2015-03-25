%PREAMBULE
load('pima.mat');

%P1
[x_1, y_1, x_2, y_2] = split_data(x, y, 0.5);

%P2
mean_ref_x1 = mean(x_1);
std_ref_x1 = std(x_1);
[x1_normalize, mean_ref_x1, std_ref_x1] = normalize_data(x_1,mean_ref_x1,std_ref_x1);
[x2_normalize, mean_ref_x2, std_ref_x2] = normalize_data(x_2,mean_ref_x1,std_ref_x1);

%P3 (a)
[y1_pred, dist1] = knn(x1_normalize, x1_normalize, y_1, 5, []);
[err_c_x1] = eval_erreur_classif(y1_pred, y_1);
[AUC1, tpr1, fpr1]= eval_AUC_ROC(y1_pred, y_1);
%P3 (b)
[y2_pred, dist2] = knn(x2_normalize, x1_normalize, y_1, 5, []);
[err_c_x2] = eval_erreur_classif(y2_pred, y_2);
[AUC2, tpr2, fpr2]= eval_AUC_ROC(y2_pred, y_2);

%MISE EN PLACE D'UNE PROCEDURE D'APPRENTISSAGE
%Separation jeu de donn?es
[x_av, y_av, x_t, y_t] = split_data(x, y, 2/3);

%Strat?gie 1
%S1.1
[x_a, y_a, x_v, y_v] = split_data(x_av, y_av, 2/3);

%S1.2
[xa_normalize, mean_ref_xa, std_ref_xa] = normalize_data(x_a,mean(x_a),std(x_a));
[xv_normalize, mean_ref_xv, std_ref_xv] = normalize_data(x_v,mean(x_a),std(x_a));
[xt_normalize, mean_ref_xt, std_ref_xt] = normalize_data(x_t,mean(x_a),std(x_a));

%S1.3
err_c_a = zeros(1,25);
for i = 1:25
    err_c_a(i) = eval_erreur_classif(knn(xa_normalize, xa_normalize, y_a, i, []), y_a);
end
err_c_v = zeros(1,25);
for i = 1:25
    err_c_v(i) = eval_erreur_classif(knn(xv_normalize, xa_normalize, y_a, i, []), y_v);
end

%S1.4
p = 1:25;
plot(p,err_c_a);
plot(p,err_c_v);

%S1.5
[err_opt, k_opt] = min(err_c_v);
err_test = eval_erreur_classif(knn(xt_normalize, xa_normalize, y_a, k_opt, []), y_t);

%Strat?gie 2
%S2.1
[xav_normalize, mean_ref_xav, std_ref_xav] = normalize_data(x_av,mean(x_av),std(x_av));
[xt_normalize, mean_ref_xt, std_ref_xt] = normalize_data(x_t,mean(x_av),std(x_av));

%S2.2
K = [1, 5, 10, 15, 20, 25];
B = 5;
n_K = length(K);
err_a = zeros(n_K, B);
err_v = zeros(n_K, B);
for b=1:B
    [x_a, y_a, x_v, y_v, indices] = split_data_fold_CV(x_av, y_av, 5, b);
    for k=1:n_K
        [ya_pred] = knn(xa_normalize, xa_normalize, y_a, K(k), []);
        [yv_pred] = knn(xv_normalize, xa_normalize, y_a, K(k), []);
        err_a(b, k) = eval_erreur_classif(ya_pred(1), y_a);
        err_v(b, k) = eval_erreur_classif(yv_pred(1), y_v);
    end
end