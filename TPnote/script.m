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

[x_av, y_av, x_t, y_t] = split_data(X, Z, 0.8);

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

class = 1:15;
n_c = length(class);
err_class = zeros(1,n_c);

for i = 1:n_c
    groups = ismember(Y, i);
    k=10;
    cvFolds = crossvalind('Kfold', groups, k);   %# get indices of 10-fold CV
    cp = classperf(groups);   %# init performance tracker

    for j = 1:k                                  %# for each fold
        testIdx = (cvFolds == j);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = svmtrain(X(trainIdx,:), groups(trainIdx), ...
                     'Autoscale',true, 'Showplot',false, 'Method','QP', ...
                     'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',1);

        %# test using test instances
        pred = svmclassify(svmModel, X(testIdx,:), 'Showplot',false);

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end
    
    err_class(i) = cp.ErrorRate;
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



