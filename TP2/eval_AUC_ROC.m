function [AUC, tpr, fpr]= eval_AUC_ROC(y_pred, y_true)

%%%
% _________________________________________________________________________
%
%   erreur_AUC_ROC.m
%   ----------------
%
%   Cette fonction calcule l'aire sous la courbe ROC en fonction de
%   reponses predites et de reponses reelles. 
%
%   On suppose que le type et les dimensions des parametres d'entree notees
%   d* ci-dessous sont correctes car verifiees avant l'appel de la
%   fonction. 

%   entrees
%   -------
%   y_pred      les reponses prediyes                   vecteur (n)
%   y_true      les vraies reponses                     vecteur (n*)
%
%   sortie
%   -------
%   AUC         l'aire sous la courbe ROC               scalaire ([0, 1])
%   tpr         le taux de vrais positifs               scalaire ([0, 1])
%   fpr         le taux de faux positifs                scalaire ([0, 1])
% _________________________________________________________________________



y_true = (y_true > 0);
[y_pred, ind] = sort(y_pred);
y_true = y_true(ind);

fpr = cumsum(y_true)/sum(y_true);
tpr = cumsum(1-y_true)/sum(1-y_true);
tpr = [0 ; tpr ; 1];
fpr = [0 ; fpr ; 1];

n = size(tpr, 1);

AUC = sum((fpr(2:n) - fpr(1:n-1)).*(tpr(2:n)+tpr(1:n-1)))/2;


