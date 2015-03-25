function [x, mean_ref, std_ref] = ...
    normalize_data(x, mean_ref, std_ref, tolerance)

%%%
% _________________________________________________________________________
%
%   normalize_data.m
%   ------------------
%
%   normalize x according to mean_ref and std_ref
%
%   inputs
%   ------
%   x           the data                                matrix (n, d)   
%   mean_ref    an empirical mean                       vector (assume: d)  
%   mean_std    an empirical standart deviation         vector (assume: d)
%
%   outputs
%   -------
%   x           the normalized data                     matrix (n, d)           
%   mean_ref    the emp. mean used to normalize         vector (d)                  
%   std_ref     the emp. std used to normalize          vector (d)    
%
%   usage
%   -----
%   [x_train, mean_x_train, std_x_train] = normalize_data(x_train);
%   [x_test] = normalize_data(x_test, mean_x_train, std_x_train);
% _________________________________________________________________________


% --> normalize data
% --------------------

if nargin < 4
    tolerance = 1e-5;
end
  
if nargin < 2 %|| isempty(mean_ref) || isempty(std_ref)
    mean_ref = mean(x);
    std_ref = std(x, 1);
    ind_zeros = find(abs(std_ref) < tolerance);

    if ~isempty(ind_zeros)
        std_ref(ind_zeros) = 1;
    end;
else
    
end

n_ref = size(x, 1);

x = x - repmat(mean_ref, n_ref, 1);
x = x ./ repmat(std_ref, n_ref, 1);