function [y_pred, dist] = knn(x_pred, x_ref, y_ref, nb_voisins, dist)

%%%
% _________________________________________________________________________
%
%   knn.m
%   -----
%
%   Cette fonction applique l'algo des k (nb_voisins) plus proches voisins
%   sur x_pred en fonction des informations disponibles dans le jeu de
%   donnees (x_ref, y_ref). 
%
%   On suppose que le type et les dimensions des parametres d'entree notees
%   d* ci-dessous sont correctes car verifiees avant l'appel de la
%   fonction.  
%   
%   entrees
%   -------
%   x_pred      jeu de donnees sur lequel           matrice (n_pred, d) 
%               on veut predire les reponses                                              
%   x_ref       observations de reference           matrice (n_ref, d*)
%   y_ref       reponses liees a x_ref              vecteur (n_ref*)
%   nb_voisins  nombre de voisins utilises          scalaire
%                
%   dist        une distance precalculee            matrice (n_ref*, n_ref*)
%
%   sorties
%   -------
%   y_pred   	les reponses predites               vecteur (n_pred)
%   dist        la distance utilisee                matrice (n_ref, n_ref)
% _________________________________________________________________________

% essayez de comprendre la fonction et commentez la

code_classe = (unique(y_ref))';
nb_classe = length(code_classe);

n_pred = size(x_pred,1);
n_ref = size(x_ref,1);
y_pred = 0 * ones(n_pred,1);

if isempty(dist) 
    for i = 1:n_pred
        for j = 1:n_ref
            % distance euclidienne
            tmp = (x_pred(i, :) - x_ref(j, :));
            dist(i, j) = tmp * tmp';
        end
    end
end

for i = 1:n_pred
    [aux, I] = sort(dist(i, :));
    C = y_ref(I);
    classe_ppv = C(1:nb_voisins);
    nc = 0*ones(nb_classe, 1);
    
    for j = 1:nb_voisins
        ind = find(code_classe == classe_ppv(j));
        nc(ind) = nc(ind)+1;    
    end
    
    [val, aff] = max(nc);
    y_pred(i) = code_classe(aff);
end;
