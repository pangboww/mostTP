function [K] = gram_matrix(X_train, X_other, kernel_type, kernel_param)

%%%
% _________________________________________________________________________
%
%   gram_matrix
%   -----------
%   
%   inputs
%   ------
%   X_train         training set (inputs)               n_train x d
%   X_other         validation or test set              n_other x d
%   kernel_type     1 = polynomial kernel               {1, 2}
%                       <X_train, X_other>^d
%                   2 = gaussian kernel
%                       exp(-1/(2*s^2)||x - x'||^2)
%   kernel_param    d for the polynomial kernel         1 x 1
%                   s for the gaussian kernel
%
%   output
%   ------
%   K               the kernel matrix                   n_train x n_other
% _________________________________________________________________________

switch kernel_type
    case 1
        K = X_train * X_other';
        K = K.^kernel_param;
    case 2
        metric = diag(1./kernel_param^2);
        ps = X_train * metric * X_other'; 
        [n_ps, d_ps] = size(ps);
        norm_X_train = sum(X_train.^2 * metric, 2);
        norm_X_other = sum(X_other.^2 * metric, 2);
        ps = -2 * ps + repmat(norm_X_train,1, d_ps) + ...
            repmat(norm_X_other',n_ps, 1) ; 
        K = exp(-ps); 
    otherwise
        error('gram_matrix : type de noyau');
end 


