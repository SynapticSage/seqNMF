function [W, H, cost,loadings,power] = seqNMF_gpu(X, varargin)
%
% USAGE: 
%
% [W, H, cost, loadings, power] = seqNMF(X, ...    % X is the data matrix
%       'K', 10, 'L', 20, 'lambda', .1, ...        % Other inputs optional
%       'W_init', W_init, 'H_init', H_init, ...
%       'showPlot', 1, 'maxiter', 20, 'tolerance', -Inf, 'shift', 1, ... 
%       'lambdaL1W', 0, 'lambdaL1H', 0, ...
%       'lambdaOrthoH', 0, 'lambdaOrthoW', 0, 'M', M)
%
% ------------------------------------------------------------------------
% DESCRIPTION:
%
%   Factorizes the NxT data matrix X into K factors 
%   Factor exemplars are returned in the NxKxL tensor W
%   Factor timecourses are returned in the KxT matrix H
%
%                                    ----------    
%                                L  /         /|
%                                  /         / |
%        ----------------         /---------/  |          ----------------
%        |              |         |         |  |          |              |
%      N |      X       |   =   N |    W    |  /   (*)  K |      H       |           
%        |              |         |         | /           |              |
%        ----------------         /----------/            ----------------
%               T                      K                         T
% See paper: 
%   XXXXXXXXXXXXXXXXX
%
% ------------------------------------------------------------------------
%
% INPUTS:
%
% Name              Default                             Description
%  X                                                    Data matrix (NxT) to factorize
% 'K'               10                                  Number of factors
% 'L'               100                                 Length (timebins) of each factor exemplar
% 'lambda'          .001                                Regularization parameter
% 'W_init'          max(X(:))*rand(N,K,L)               Initial W
% 'H_init'          max(X(:))*rand(K,T)./(sqrt(T/3))    Initial H (rows have norm ~1 if max(data) is 1)
% 'showPlot'        1                                   Plot every iteration? no=0
% 'maxiter'         100                                 Maximum # iterations to run
% 'tolerance'       -Inf                                Stop if improved less than this;  Set to -Inf to always run maxiter
% 'shift'           1                                   Shift factors to center; Helps avoid local minima
% 'lambdaL1W'       0                                   L1 sparsity parameter; Increase to make W's more sparse
% 'lambdaL1H'       0                                   L1 sparsity parameter; Increase to make H's more sparse
% 'W_fixed'         0                                   Fix W during the fitting proceedure   
% 'SortFactors'     1                                   Sort factors by loadings
% 'lambdaOrthoH'    0                                   ||HSH^T||_1,i~=j; Encourages events-based factorizations
% 'lambdaOrthoW'    0                                   ||Wflat^TWflat||_1,i~=j; ; Encourages parts-based factorizations
% 'useWupdate'      1                                   Wupdate for cross orthogonality often doesn't change results much, and can be slow, so option to remove  
% 'M'               ones(N,T)                           Masking matrix if excluding a random test set from the fit
% ------------------------------------------------------------------------
% OUTPUTS:
%
% W                         NxKxL tensor containing factor exemplars
% H                         KxT matrix containing factor timecourses
% cost                      1x(#Iterations+1) vector containing 
%                               reconstruction error at each iteration. 
%                               cost(1) is error before 1st iteration.
% loadings                  1xK vector containing loading of each factor 
%                               (Fraction power in data explained by each factor)
% power                     Fraction power in data explained 
%                               by whole reconstruction
%
%                           Note, if doing fit with masked (held-out) data,
%                               the cost and power do not include masked
%                               (M==0) test set elements
% ------------------------------------------------------------------------
% CREDITS:
%   Emily Mackevicius and Andrew Bahle, 2/1/2018
%
%   Original CNMF algorithm: Paris Smaragdis 2004
%   (https://link.springer.com/chapter/10.1007/978-3-540-30110-3_63)
%   Adapted from NMF toolbox by Colin Vaz 2015 (http://sail.usc.edu)
%
%   Please cite our paper: 
%       https://www.biorxiv.org/content/early/2018/03/02/273128
% ------------------------------------------------------------------------
%  GPU version
%
%  Any of the major matrices are cast into gpuArray. Additionally, data
%  on the gpu is cast to single so that one can fit larger matrices.
%  
%  Runs about 5x faster on a good graphics card. Tested on a
%  Quadro P6000.

%
%% parse function inputs

% Check that we have non-negative data
if min(X(:)) < 0
    error('Negative values in data!');
end

% Parse inputs
[X,N,T,K,L,params] = parse_seqNMF_params(X, varargin);

%% initialize
W = params.W_init;
H = params.H_init;

Xhat = gpuArray(single(helper.reconstruct(W, H))); 
mask = find(params.M == 0); % find masked (held-out) indices 
X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit

smoothkernel = gpuArray(single(ones(1,(2*L)-1)));  % for factor competition
smallnum = max(X(:))*1e-6; 
lasttime = 0;

% Calculate initial cost
cost = gpuArray(single(zeros(params.maxiter+1, 1)));
cost(1) = sqrt(mean((X(:)-Xhat(:)).^2));

for iter = 1 : params.maxiter
    % Stopping criteria... Stop if reach maxiter or if change in cost function is less than the tolerance
    if (iter == params.maxiter) || ((iter>5) && (cost(iter+1)+params.tolerance)>mean(cost((iter-5):iter)))
        cost = cost(1 : iter+1);  % trim vector
        lasttime = 1; 
        if iter>1
            params.lambda = gpuArray(0); % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
        end
    end
    
    % Compute terms for standard CNMF H update 
    WTX = zeros(K, T, 'gpuArray');
    WTXhat = zeros(K, T, 'gpuArray');
    for l = 1 : L
        X_shifted = circshift(X,[0,-l+1]); 
        Xhat_shifted = circshift(Xhat,[0,-l+1]); 
        WTX = WTX + W(:, :, l)' * X_shifted;
        WTXhat = WTXhat + W(:, :, l)' * Xhat_shifted;
    end   
         
    % Compute regularization terms for H update
    if params.lambda>0
        dRdH = params.lambda.*(~eye(K))*conv2(WTX, smoothkernel, 'same');  
    else 
        dRdH = 0; 
    end
    if params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH * (~eye(K))*conv2(H, smoothkernel, 'same');
    else
        dHHdH = gpuArray(0);
    end
    dRdH = dRdH + params.lambdaL1H + dHHdH; % include L1 sparsity, if specified
    
    % Update H
    H = H .* WTX ./ (WTXhat + dRdH +eps);
        
    % Shift to center factors
    if params.shift
        [W, H] = helper.shiftFactors(W, H);  
        W = W+smallnum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    
    % Renormalize so rows of H have constant energy
    norms = sqrt(sum(H.^2, 2))';
    H = diag(1 ./ (norms+eps)) * H;
    for l = 1 : L
        W(:, :, l) = W(:, :, l) * diag(norms);
    end 
    
    if ~params.W_fixed
    % Update each Wl separately
        Xhat = helper.reconstruct(W, H); 
        mask = find(params.M == 0); % find masked (held-out) indices 
        X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
        if params.lambdaOrthoW>0
            Wflat = sum(W,3);
        end
        if params.lambda>0 && params.useWupdate
            XS = conv2(X, smoothkernel, 'same'); 
        end
        for l = 1 : L % could parallelize to speed up for long L
            % Compute terms for standard CNMF W update
            H_shifted = circshift(H,[0,l-1]);
            XHT = X * H_shifted';
            XhatHT = Xhat * H_shifted';

            % Compute regularization terms for W update
            if params.lambda>0 && params.useWupdate; % Often get similar results with just H update, so option to skip W update
                dRdW = params.lambda.*XS*(H_shifted')*(~eye(K)); 
            else
                dRdW = 0;
            end
            if params.lambdaOrthoW>0
                dWWdW = params.lambdaOrthoW*Wflat*(~eye(K));
            else
                dWWdW = 0;
            end
            dRdW = dRdW + params.lambdaL1W + dWWdW; % include L1 and Worthogonality sparsity, if specified
            % Update W
            W(:, :, l) = W(:, :, l) .* XHT ./ (XhatHT + dRdW + eps);
        end
    end
    % Calculate cost for this iteration
    Xhat = helper.reconstruct(W, H);    
    mask = find(params.M == 0); % find masked (held-out) indices 
    X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    cost(iter+1) = sqrt(mean((X(:)-Xhat(:)).^2));

    % Plot to show progress
    if params.showPlot 
        SimpleWHPlot(W, H, Xhat,0); 
        title(sprintf('iteration #%i',iter));
        drawnow
    end
    
    if lasttime
        break
    end

end
   
% Undo zeropadding by truncating X, Xhat and H
X = X(:,L+1:end-L);
Xhat = Xhat(:,L+1:end-L);
H = H(:,L+1:end-L);

% Compute explained power of whole reconstruction and each factor
power = (sum(X(:).^2)-sum((X(:)-Xhat(:)).^2))/sum(X(:).^2);  % fraction power explained by whole reconstruction
[loadings,ind] = sort(helper.computeLoadingPercentPower(gather(X),gather(W),gather(H)),'descend'); % fraction power explained by each factor

% sort factors by loading power
if params.SortFactors
    W = W(:,ind,:);
    H = H(ind,:);
end

% Cast back to CPU
W = gather(W);
H = gather(H);
power = gather(power);
cost = gather(cost);
loadings = gather(loadings);

    function [X,N,T,K,L,params] = parse_seqNMF_params(X, inputs);
        % parse inputs, set unspecified parameters to the defaults
        
        % Get data dimensions
        [N, T] = size(X);

        p = inputParser; % 
        %USAGE: addOptional(p,'parametername',defaultvalue);
        addOptional(p,'K',10);
        addOptional(p,'L',100);
        addOptional(p,'lambda',.001);
        addOptional(p,'showPlot',1);
        addOptional(p,'maxiter',100);
        addOptional(p,'tolerance',-Inf);
        addOptional(p,'shift',1);
        addOptional(p,'lambdaL1W',0);
        addOptional(p,'lambdaL1H',0);
        addOptional(p,'W_fixed',0);
        addOptional(p,'W_init', nan); % depends on K--initialize post parse
        addOptional(p,'H_init', nan); % depends on K--initialize post parse
        addOptional(p,'SortFactors', 1); % sort factors by loading?
        addOptional(p,'lambdaOrthoW',0); % for this regularization: ||Wflat^TWflat||_1,i~=j
        addOptional(p,'lambdaOrthoH',0); % for this regularization: ||HSH^T||_1,i~=j
        addOptional(p,'useWupdate',1); % W update for cross orthogonality often doesn't change results much, and can be slow, so option to skip it 
        addOptional(p,'M',nan); % Masking matrix: default is ones; set elements to zero to hold out as masked test set
       
        parse(p,inputs{:});
        L = gpuArray(single(p.Results.L)); 
        K = gpuArray(single(p.Results.K)); 
        params = p.Results; 
        
        % zeropad data by L
        X = gpuArray(single([zeros(N,L),X,zeros(N,L)]));
        [N, T] = size(X);
        N = gpuArray(single(N));
        T = gpuArray(single(T));

        % initialize W_init and H_init, if not provided
        if isnan(params.W_init)
            params.W_init = max(X(:))*rand(N, K, L, 'single', 'gpuArray');
        end
        if isnan(params.H_init)
            params.H_init = max(X(:))*rand(K,T, 'single','gpuArray')./(sqrt(T/3)); % normalize so frobenius norm of each row ~ 1
        else
            params.H_init = single([zeros(K,L,'single','gpuArray'),params.H_init,zeros(K,L,'single','gpuArray')]);
        end
        if isnan(params.M)
            params.M = ones(N,T,'single','gpuArray');
        else
            params.M = [ones(N,L,'single','gpuArray'),params.M,ones(N,L,'single','gpuArray')];
        end

        % Cast any terms that might be accessed in the loop and
        % part of any matrix computations
        params.lambda       = gpuArray(single(params.lambda));
        params.lambdaL1W    = gpuArray(single(params.lambdaL1W));
        params.lambdaL1H    = gpuArray(single(params.lambdaL1H));
        params.lambdaOrthoW = gpuArray(single(params.lambdaOrthoW));
        params.lambdaOrthoW = gpuArray(single(params.lambdaOrthoH));
    end
end
