clear;clc;

load 'page_blocks.mat'
% Definition de differents parametres
ratio = [1, 2, 3, 4, 5, 10, 50, 75, 100];
param_L = size(X_train, 1) ./ ratio;
for p = 1:length(param_L)
    L = param_L(p);
    template = templateTree('MinLeaf', L);
    method = 'AdaBoostM2';
    % Creation d'un modele e partir des donnees d'apprentissage
    model = fitensemble(X_train, y_train, method, 50, template);
    % Prediction sur des donnees (X_train ou une autre matrice de donnees)
    y_hat_train=predict(model, X_train);
    y_hat_valid=predict(model,X_valid);
    y_hat_test =predict(model,X_test);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_train = confusionmat(y_train,y_hat_train) ./ size(X_train, 1);
    conf_matrix_valid = confusionmat(y_valid,y_hat_valid) ./ size(X_valid, 1);
    conf_matrix_test = confusionmat(y_test,y_hat_test) ./ size(X_test, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_train));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_valid));
    erreur_test(p) = 1 - sum(diag(conf_matrix_test));
end

plot(ratio,erreur_train,'bx-'); hold on;
plot(ratio,erreur_valid,'go-'); hold on;
plot(ratio,erreur_test,'rs-');

% Subspace et k plus proches voisins
% Definition de differents parametres
param_K = (1:10);
for p = 1:length(param_K)
    K = param_K(p);
    template = templateKNN('Numneighbors', K);
    method = 'Subspace';
    model = fitensemble(X_train, y_train, method, 50, template);
    % Prediction sur des donnees (X_train ou une autre matrice de donnees)
    y_hat_train=predict(model, X_train);
    y_hat_valid=predict(model,X_valid);
    y_hat_test =predict(model,X_test);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_train = confusionmat(y_train,y_hat_train) ./ size(X_train, 1);
    conf_matrix_valid = confusionmat(y_valid,y_hat_valid) ./ size(X_valid, 1);
    conf_matrix_test = confusionmat(y_test,y_hat_test) ./ size(X_test, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_train));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_valid));
    erreur_test(p) = 1 - sum(diag(conf_matrix_test));    
end

plot(param_K,erreur_train,'bx-'); hold on;
plot(param_K,erreur_valid,'go-'); hold on;
plot(param_K,erreur_test,'rs-');

% Nombre d'apprenants par cycly
% Definition de differents parametres
param_nl = [200, 500, 1000];

for p = 1:length(param_nl)
    % Creqtion d'un modele pour un parametre nlear specifique
    method = 'AdaboostM2';
    nl = param_nl(p);
    model = fitensemble(X_train, y_train, method, nl, 'Tree');
    % Prediction sur des donnees (X_train ou une autre matrice de donnees)
    y_hat_train=predict(model, X_train);
    y_hat_valid=predict(model, X_valid);
    y_hat_test =predict(model, X_test);
    % Creation d'une matrice de confusion pour calculer l'erreur
    conf_matrix_train = confusionmat(y_train,y_hat_train) ./ size(X_train, 1);
    conf_matrix_valid = confusionmat(y_valid,y_hat_valid) ./ size(X_valid, 1);
    conf_matrix_test = confusionmat(y_test,y_hat_test) ./ size(X_test, 1);

    % Calcul de l'erreur
    erreur_train(p) = 1 - sum(diag(conf_matrix_train));
    erreur_valid(p) = 1 - sum(diag(conf_matrix_valid));
    erreur_test(p) = 1 - sum(diag(conf_matrix_test)); 
end

plot(loss(model, X_train, y_train, 'mode', 'cumulative'), 'bx-'); hold on;
plot(loss(model, X_valid, y_valid, 'mode', 'cumulative'), 'cs-'); hold on;
plot(loss(model, X_test, y_test, 'mode', 'cumulative'), 'ro-');
xlabel('Number of trees');
ylabel('Classification error');



% Erreur de bootstrap

