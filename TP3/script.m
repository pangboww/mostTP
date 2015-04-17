%------EX.12
clear;
clc;
lambda = 10.^(0:6);
X = randn(1000, 100);
y = sign(X(:, 1));
[K] = gram_matrix(X, X, 1, 1);
%--------initialise les variables
alpha = zeros(size(K, 1), length(lambda));
b= zeros(length(lambda), 1);
err_c = zeros(length(lambda), 1);
t= zeros(length(lambda), 1);

for i=1:length(lambda)
   [alpha(:,i), b(i)]= optimize_svm(K,y,lambda(i));
   
   [err_c(i)] = eval_erreur_classif(sign(alpha(:,i)'*K+b(i))',y);
    % nombre de lignes non-nulles dans alpha(:,i)
   t(i) = sum(alpha(:,i) ~= 0);
end

%------EX.345
clear;
clc;

load test_tp.mat
ind_pos = find(y == 1);
ind_neg = find(y == -1);
plot(X(ind_pos, 1), X(ind_pos, 2), 'xr');
hold on;
plot(X(ind_neg, 1), X(ind_neg, 2), '.b');


% %---------ex.3,4,5
lambda = 10.^(-3:3);
[K_1] = gram_matrix(X, X, 2, sqrt(2));

alpha_1 = zeros(size(K_1, 1), length(lambda));
b_1 = zeros(length(lambda), 1);
err_c2 = zeros(length(lambda), 1);
t_1 = zeros(length(lambda), 1);

for i=1:length(lambda)
   [alpha_1(:, i), b_1(i)]= optimize_svm(K_1,y,lambda(i));
   
   [err_c2(i)] = eval_erreur_classif(sign(alpha_1(:,i)'*K_1+b_1(i))',y);
   
   t_1(i) = sum(alpha_1(:,i) ~= 0);
   
end

clf;
figure(1);
semilogx(lambda, t_1, 'r-');
figure(2);
semilogx(lambda, err_c2, 'b-');









% ----ex.6

clear; clc;

X_train = importdata('banana_train_data.txt', ' ');
y_train = importdata('banana_train_labels.txt', ' ');
X_valid = importdata('banana_valid_data.txt', ' ');
y_valid = importdata('banana_valid_labels.txt', ' ');
X_test  = importdata('banana_test_data.txt', ' ');
y_test  = importdata('banana_test_labels.txt', ' ');

% ----ex.6
%-----initialisations
lambda = 10.^(-3:3);
nb_lambda = length(lambda);
sigma = 1:1:10;
nb_sigma = length(sigma);

train_err = zeros(nb_lambda, nb_sigma);
valid_err = zeros(nb_lambda, nb_sigma);

for ind_lambda = 1: nb_lambda
    for ind_sigma = 1:nb_sigma
        [K_train] = gram_matrix(X_train, X_train, 2, sigma(ind_sigma));
        [alpha, b] = optimize_svm(K_train, y_train, lambda(ind_lambda));
        y_train_pred = sign(alpha'*K_train+b);
        [train_err(ind_lambda, ind_sigma)] = eval_erreur_classif(y_train_pred', y_train);
        
        [K_valid] = gram_matrix(X_train, X_valid, 2, sigma(ind_sigma));
        y_valid_pred= sign(alpha'*K_valid+b); 
        [valid_err(ind_lambda, ind_sigma)] = eval_erreur_classif(y_valid_pred', y_valid);
    end
end

%----trouver le minimum dans valid_err 
[M,I] = min(valid_err(:));
[ind_lambda_opt, ind_sigma_opt] = ind2sub(size(valid_err),I);

[alpha, b] = optimize_svm(K_train, y_train, lambda(ind_lambda_opt));
[K_test] = gram_matrix(X_train, X_test, 2, sigma(ind_sigma_opt));
y_test_pred = sign(alpha'*K_test+b);
test_err = eval_erreur_classif(y_test_pred', y_test);
            
sprintf('L"erreur de test vaut : %2.2f', test_err);





%%%%%%%%%%-ex.7
clear;
clc;

X_train = importdata('banana_train_data.txt', ' ');
y_train = importdata('banana_train_labels.txt', ' ');
X_valid = importdata('banana_valid_data.txt', ' ');

y_valid = importdata('banana_valid_labels.txt', ' ');
X_test  = importdata('banana_test_data.txt', ' ');
y_test  = importdata('banana_test_labels.txt', ' ');

X_tv = [X_train; X_valid];
y_tv = [y_train; y_valid];

%------------ex 8

lambda = 10.^(-3:3);
nb_lambda = length(lambda);
sigma = 1:1:10;
nb_sigma = length(sigma);
B = 5;
train_err = zeros(nb_lambda, nb_sigma);
valid_err = zeros(nb_lambda, nb_sigma);

for b = 1:B
    for ind_sigma = 1:nb_sigma
        [X_t, Y_t, X_v, Y_v] = split_data_fold_CV(X_tv, y_tv, B, b);
        [K_t] = gram_matrix(X_t, X_t, 2, sigma(ind_sigma));
        [K_v] = gram_matrix(X_t, X_v, 2, sigma(ind_sigma));
        
        for ind_lambda = 1:nb_lambda
             [alpha, c] = optimize_svm(K_t, Y_t, lambda(ind_lambda));
             y_t_pred = sign(alpha'*K_t+c); 
             [train_err(ind_lambda, ind_sigma)] = (1/B)*eval_erreur_classif(y_t_pred', Y_t);
             y_v_pred = sign(alpha'*K_v+c);
             [valid_err(ind_lambda, ind_sigma)] = (1/B)*eval_erreur_classif(y_v_pred', Y_v);
        end
    end
end
 
[M,I] = min(valid_err(:));
[ind_lambda_opt, ind_sigma_opt] = ind2sub(size(valid_err),I);

%-----test
[K_test] = gram_matrix(X_tv, X_test, 2, sigma(ind_sigma_opt));
% ajout
[K_tv] = gram_matrix(X_tv, X_tv, 2, sigma(ind_sigma_opt));
[alpha_opt, c_opt] = optimize_svm(K_tv, y_tv, lambda(ind_lambda_opt));

y_test_pred = sign(alpha_opt'*K_test+c_opt);
test_err = eval_erreur_classif(y_test_pred', y_test);