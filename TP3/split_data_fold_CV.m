function [x_a, y_a, x_v, y_v, indices] = ...
                                split_data_fold_CV(x, y, B_blocs, num_bloc)

%%%
% _________________________________________________________________________
%
%   split_data_fold_CV.m
%   --------------------
%
%   Fonction permettant de construire l'ensemble des B blocs (folds), a
%   utiliser dans un script plus general de validation croisee.
%
%   exemple
%   -------
%
%   K = [1, 5, 10, 15, 20, 25];
%   B = 5;
%
%   n_K = length(K);
%
%   err_a = zeros(n_K, B);
%   err_v = zeros(n_K, B);
%
%
%   for b=1:B
%       [x_a, y_a, x_v, y_v, indices] = ...
%           split_data_fold_CV(x_av, y_av, 5, b);
%
%       for k=1:n_K
%           [y_pred] = knn(...) avec ... les ensembles definis precedemment
%                               avec le nombre de voisins teste = K(k)
%           err_a(b, k) = ...
%           err_v(b, k) = ...
%       end
%   end
%
% _________________________________________________________________________

if num_bloc > B_blocs
    error('split_data_fold_CV: num_bloc > a B_blocs')
end

code_classe = unique(y);
C = length(code_classe);

% Intitaliser des tableaux de cette maniere, c'est *mal* mais on verra plus
% tard comment faire autrement. Eventuellement.
ind_a = []; 
ind_v = [];

% Repartit de facon equilibree le nombre d'individus appartenant a la
% classe c dans les differents blocs. 
for c=1:C
    ind_classe_c = find(y == code_classe(c));
    nb_ind_c = length(ind_classe_c);
    taille_portion_c = round(nb_ind_c / B_blocs); % longueur de chaque portion
    
    %nb_ind_c = nb_ind_c + length(ind_classe_c);
    
    ind_v_classe_c = ind_classe_c(taille_portion_c * (num_bloc-1) + 1 ...
        : min(taille_portion_c * num_bloc, nb_ind_c) );
    ind_v = [ind_v; ind_v_classe_c];
    ind_a_classe_c = setdiff(ind_classe_c, ind_v_classe_c);
    ind_a = [ind_a; ind_a_classe_c];
end

x_a  = x(ind_a, :);   
y_a = y(ind_a);

x_v = x(ind_v, :);  
y_v = y(ind_v);

indices.a = ind_a;
indices.v = ind_v;

