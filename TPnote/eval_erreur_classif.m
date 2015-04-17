function [err_c] = eval_erreur_classif(y_pred, y_true)

%%%
% _________________________________________________________________________
%
%   eval_erreur_classif.m
%   ---------------------
%
%   Cette fonction calcule l'erreur de classification  en fonction de
%   reponses predites et de reponses reelles. 
%   Elle pourrait etre adaptee pour associer un cout a chaque type
%   d'erreur.  
%   
%   On suppose que le type et les dimensions des parametres d'entree notees
%   d* ci-dessous sont correctes car verifiees avant l'appel de la
%   fonction. 

%   entrees
%   -------
%   y_pred      les reponses predites                   vecteur (n)
%   y_true      les vraies reponses                     vecteur (n*)
%
%   sortie
%   -------
%   err_c       l'erreur de classification              scalaire ([0, 1])
% _________________________________________________________________________

err_c = sum(y_pred ~= y_true) / length(y_true);

end

