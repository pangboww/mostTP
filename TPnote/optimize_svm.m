function [alpha, b] = optimize_svm(K, y, lambda)

%%%
% _________________________________________________________________________
%
%   optimize_svm
%   ------------
%   
%   inputs
%   ------
%   K               kernel matrix (inputs)                  n x n
%   y               labels (outputs)                        n x 1
%   lambda          regularization parameter                1 x 1
%
%
%   output
%   ------
%   alpha           sparsified vector of svm                n x 1
%   b               biais of svm                            1 x 1
% _________________________________________________________________________

max_iter = 10;
n = size(K, 1);
alpha_b = zeros(n+1, 1); %last one is b
   
if n >= 1000,
    permut = randperm(n);
    sv = permut(1:(floor(n/2)));
    [small_beta, small_b] = optimize_svm(K(sv, sv), y(sv), lambda);
    alpha_b(sv) = small_beta; 
    alpha_b(end) = small_b;
else
    sv = (1:n);
end

nb_iter = 0;
goon = 1;
cste = mean(diag(K));
    
while goon,
    old_sv = sv;
    
    out = 1 - y .* (K*alpha_b(1:end-1) + alpha_b(end));
    sv = find(out >= 0);
    %obj = ((alpha_b' * Kb) + lambda * sum(max(0, out).^2))/2;  

    nb_iter = nb_iter + 1;
    
    % conditions d'arret
    if (nb_iter > 1) && isempty(setxor(sv, old_sv))
        break;
    end
    
    if (nb_iter > max_iter)
        disp('Maximum number of Newton steps reached'); 
        % break;
    end
    
    % newton step
    H = K(sv, sv) + lambda * eye(length(sv));
    H(end+1, :) = cste;
    H(:, end+1) = cste;
    H(end, end) = 0;
       
    alpha_b = zeros(n+1, 1);
    alpha_b([sv; end]) = H\[y(sv);0];
    alpha_b(end) = alpha_b(end) * cste;   
end

alpha = alpha_b(1:end-1);

% sparsify final alpha
alpha(abs(alpha) <= 1e-6) = 0;

b = alpha(end);
