function [x_1, y_1, x_2, y_2] = split_data(x, y, ratio)

%%%
% _________________________________________________________________________
%
%   split_data.m
%   ------------
%
%   Cette fonction s?pare en deux le jeux de donnees (x, y) en fonction du
%   ratio 'ratio' et en tenant compte de la repartition des classes au sein
%   de chaque jeu de donnees.
%   
%   entrees
%   -------
%   x       matrice des observations                        matrice (n, d)
%   y       vecteur de reponses associees ? x               vecteur (n)
%
%   sorties
%   -------
%   x_1     sous-matrice 1 des observations                 matrice (n1, d)
%   y_1     sous-vecteur 1 de reponses                      vecteur (n1)
%   x_2     sous-matrice 2 des observations                 matrice (n1, d)
%   y_2     sous-vecteur 2 de reponses                      vecteur (n1)
% _________________________________________________________________________


code_classe = unique(y);

% intitaliser des tableaux de cette maniere, c'est *mal* mais on verra
% eventuellement plus tard comment faire autrement
x_1 = [];
y_1 = [];
x_2 = [];
y_2 = [];
C = length(code_classe);

for c = 1:C
    indice_classe = find(y == code_classe(c));
    nb_i_classe  = length(indice_classe);
    
    aux = randperm(nb_i_classe);
    aux_1 = aux(1:ceil(ratio * nb_i_classe));
    aux_2 = aux(ceil(ratio * nb_i_classe) + 1:end);
    
    x_1 = [x_1; x(indice_classe(aux_1), :)];
    y_1 = [y_1; y(indice_classe(aux_1))];
    
    x_2 = [x_2; x(indice_classe(aux_2), :)];
    y_2 = [y_2; y(indice_classe(aux_2))];
end