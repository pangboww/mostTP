%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      k plus proches voisins                         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------- Euclidienne --------%

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
        
        conf_matrix_a = confusionmat(y_a,y_ap) ./ size(x_a, 1);
        conf_matrix_v = confusionmat(y_v,y_vp) ./ size(x_v, 1);
        
        err_a(k,b) = 1 - sum(diag(conf_matrix_a));
        err_v(k,b) = 1 - sum(diag(conf_matrix_v));
    end
end

plot(K,mean(err_a,2),'r');hold on;
plot(K,mean(err_v,2),'b');
legend('training error','validation error','Location','southeast');

[val_mmin, ind_min] = min(mean(err_v, 2));
mdl = fitcknn(x_av,y_av,'NumNeighbors',K(ind_min),'Distance','euclidean');
y_tp = predict(mdl,x_t);
conf_matrix_t = confusionmat(y_t,y_tp) ./ size(x_t,1);
err_t = 1 - sum(diag(conf_matrix_t));


% ----------- Cityblock ------------%

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
        
        conf_matrix_a = confusionmat(y_a,y_ap) ./ size(x_a, 1);
        conf_matrix_v = confusionmat(y_v,y_vp) ./ size(x_v, 1);
        
        err_a(k,b) = 1 - sum(diag(conf_matrix_a));
        err_v(k,b) = 1 - sum(diag(conf_matrix_v));
    end
end


plot(K,mean(err_a,2),'r');hold on;
plot(K,mean(err_v,2),'b');
legend('training error','validation error','Location','southeast');

[val_mmin, ind_min] = min(mean(err_v, 2));
mdl = fitcknn(x_av,y_av,'NumNeighbors',K(ind_min),'Distance','cityblock');
y_tp = predict(mdl,x_t);
conf_matrix_t = confusionmat(y_t,y_tp) ./ size(x_t,1);
err_t = 1 - sum(diag(conf_matrix_t));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                SVM                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

load('Yale_Faces.mat');

[x_av, y_av, x_t, y_t] = split_data(X, Y, 0.8);

lambda = 10.^(-3:3);
nb_lambda = length(lambda);
sigma = 1:1:10;
nb_sigma = length(sigma);
B = 5;
train_err = zeros(nb_lambda, nb_sigma);
valid_err = zeros(nb_lambda, nb_sigma);

for b = 1:B
    for ind_sigma = 1:nb_sigma
        [x_a, y_a, x_v, y_v] = split_data_fold_CV(x_av, y_av, B, b);
        [k_a] = gram_matrix(x_a, x_a, 2, sigma(ind_sigma));
        [k_v] = gram_matrix(x_a, x_v, 2, sigma(ind_sigma));
        
        for ind_lambda = 1:nb_lambda
            mdl = fitcsvm(x_a,y_a,'Alpha',
            
            [alpha, c] = optimize_svm(k_a, y_a, lambda(ind_lambda));
            y_a_pred = sign(alpha'*k_a+c); 
%              [train_err(ind_lambda, ind_sigma)] = (1/B)*eval_erreur_classif(y_a_pred', y_a);
            y_v_pred = sign(alpha'*k_v+c);
%              [valid_err(ind_lambda, ind_sigma)] = (1/B)*eval_erreur_classif(y_v_pred', y_v);
            conf_matrix_a = confusionmat(y_a,y_a_pred) ./ size(x_a, 1);
            conf_matrix_v = confusionmat(y_v,y_a_pred) ./ size(x_v, 1);
        
            train_err(ind_lambda, ind_sigma) = 1 - sum(diag(conf_matrix_a));
            valid_err(ind_lambda, ind_sigma) = 1 - sum(diag(conf_matrix_v));

        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                             Boosting                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear;
clc;

load('Yale_Faces.mat');

[x_av, y_av, x_t, y_t] = split_data(X, Y, 0.8);
[x_a, y_a, x_v, y_v] = split_data(x_av, y_av, 0.5);

ratio = [4, 5, 10, 50];
param_L = size(x_a, 1) ./ ratio;
for p = 1:length(param_L)
    L = param_L(p);
    template = templateTree('MinLeaf', L);
    method = 'AdaBoostM2';
    % Creation d'un modele e partir des donnees d'apprentissage
    model = fitensemble(x_a, y_a, method, 50, template);
    % Prediction sur des donnees (x_a ou une autre matrice de donnees)
    y_hat_train=predict(model,x_a);
    y_hat_valid=predict(model,x_v);
    y_hat_test =predict(model,x_t);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_a = confusionmat(y_a,y_hat_train) ./ size(x_a, 1);
    conf_matrix_v = confusionmat(y_v,y_hat_valid) ./ size(x_v, 1);
    conf_matrix_t = confusionmat(y_t,y_hat_test) ./ size(x_t, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_a));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_v));
    erreur_test(p) = 1 - sum(diag(conf_matrix_t));
end

plot(ratio,erreur_train,'bx-'); hold on;
plot(ratio,erreur_valid,'go-'); hold on;
plot(ratio,erreur_test,'rs-');


param_K = (1:10);
for p = 1:length(param_K)
    K = param_K(p);
    template = templateKNN('Numneighbors', K);
    method = 'Subspace';
    model = fitensemble(x_a, y_a, method, 50, template);
    % Prediction sur des donnees (x_a ou une autre matrice de donnees)
    y_hat_train=predict(model, x_a);
    y_hat_valid=predict(model,x_v);
    y_hat_test =predict(model,x_t);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_a = confusionmat(y_a,y_hat_train) ./ size(x_a, 1);
    conf_matrix_v = confusionmat(y_v,y_hat_valid) ./ size(x_v, 1);
    conf_matrix_t = confusionmat(y_t,y_hat_test) ./ size(x_t, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_a));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_v));
    erreur_test(p) = 1 - sum(diag(conf_matrix_t));    
end

plot(param_K,erreur_train,'bx-'); hold on;
plot(param_K,erreur_valid,'go-'); hold on;
plot(param_K,erreur_test,'rs-');



param_nl = [200, 500, 1000];

for p = 1:length(param_nl)
    % Creqtion d'un modele pour un parametre nlear specifique
    method = 'AdaboostM2';
    nl = param_nl(p);
    model = fitensemble(x_a, y_a, method, nl, 'Tree');
    % Prediction sur des donnees (x_a ou une autre matrice de donnees)
    y_hat_train=predict(model, x_a);
    y_hat_valid=predict(model, x_v);
    y_hat_test =predict(model, x_t);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_a = confusionmat(y_a,y_hat_train) ./ size(x_a, 1);
    conf_matrix_v = confusionmat(y_v,y_hat_valid) ./ size(x_v, 1);
    conf_matrix_t = confusionmat(y_t,y_hat_test) ./ size(x_t, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_a));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_v));
    erreur_test(p) = 1 - sum(diag(conf_matrix_t)); 
end

plot(loss(model, x_a, y_a, 'mode', 'cumulative'), 'bx-'); hold on;
plot(loss(model, x_v, y_v, 'mode', 'cumulative'), 'cs-'); hold on;
plot(loss(model, x_t, y_t, 'mode', 'cumulative'), 'ro-');
xlabel('Number of trees');
ylabel('Classification error');



