% Compute the loglikelohood value and its gradient.
% Based on V - Scales are the function of some parameters
%%
function [LL, grad] = getLL3_nested()

    global incidenceFull; 
    global Gradient;
    global Op;
    global Mfull;
    global Ufull;
    global Atts;
    global Obs;     % Observation
    global nbobs;  
    global isLinkSizeInclusive;
    global lastIndexNetworkState;
    global Scale;
    global mu;

    %% Get M, U
    [lastIndexNetworkState, maxDest] = size(incidenceFull); 
    mu = getMu(Op.x);
    Mfull = getM(Op.x, isLinkSizeInclusive);
    MregularNetwork = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Ufull = getU(Op.x, isLinkSizeInclusive);
    UregularNetwork = Ufull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    %% Set initial LL value
    LL = 0;
    grad = zeros(1, Op.n);
    
    %% Initialize
    M = MregularNetwork;
    M(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    M(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    U = UregularNetwork;
    U(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    U(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    for i = 1:Op.n
        AttLc(i) =  Matrix2D(Atts(i).Value(1:lastIndexNetworkState,1:lastIndexNetworkState));
        AttLc(i).Value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        AttLc(i).Value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end

    %% Compute B matrix:
    N = size(M,1);
    b = sparse(zeros(N,1));
    b(N) = 1;
    B = sparse(zeros(N, maxDest - lastIndexNetworkState));
    B(N,:) = ones(1,maxDest - lastIndexNetworkState);
    for i = 1: maxDest - lastIndexNetworkState
        B(1:lastIndexNetworkState,i) = Mfull(:, i+lastIndexNetworkState);
    end
    
    %% Compute Z by iterative method:
    % B = B(:,1:4);
    [Z, expVokBool]   = getZ(M, B);
    if (expVokBool == 0)
            LL = OptimizeConstant.LL_ERROR_VALUE;
            grad = ones(Op.n,1);
            disp('The parameters not fesible')
            return; 
    end
    
    %% Compute V
    V = log(Z);
    V = (bsxfun(@times,Scale,V));
    
    %% Get gradient    
    gradExpV = objArray(Op.n);
    gradV = objArray(Op.n);
    MI = sparse(M); 
    MI(find(M)) = 1;
    a = Scale;
    k = 1 ./ Scale;
    kf = k(1:lastIndexNetworkState,1);
    phi = sparse((k * a') .* MI);    
    e = ones(size(M,1),1);
    
    %% Get Gradient of V
    for i = 1:Op.n
        gradVi = zeros(size(B));
        gradt = Atts(i).Value(:,lastIndexNetworkState+1 : maxDest) .* Mfull(:,lastIndexNetworkState+1 : maxDest);
        gradt = bsxfun(@times,kf,gradt);
        gradt(lastIndexNetworkState+1,:) = sparse(zeros(1,maxDest - lastIndexNetworkState));        
        gradt = sparse(gradt);
        gradt = gradt(:,1:size(B,2)) .* bsxfun(@times,k,Z);
        for d = 1: size(B,2)
            % Compute P, N
            gradM = M .* (AttLc(i).Value);
            gradM = bsxfun(@times,k,gradM);
            Zd = sparse(bsxfun(@times,Z(:,d)',MI));
            X = MI;
            X(find(MI)) =  Zd(find(MI)) .^ (phi(find(MI)));
            X = bsxfun(@times,1 ./ Z(:,d),X);
            H = M .* X;
            K =  bsxfun(@times, a ,(gradM .* X));
            gradVi(:,d) = (speye(size(M)) - H)\( K * e + gradt(:,d));  
        end
        gradV(i).value = gradVi;  
    end
    
    %% Compute gradient of Z    
    for i = 1:Op.n
        X = gradV(i).value .* Z;
        gradExpV(i).value = sparse(bsxfun(@times,k,X));
    end
    
    %% Compute the LL and gradient.
    gradVd = zeros(size(Z,1),Op.n);
    for n = 1:nbobs    
%        n
        dest = Obs(n, 1);
        orig = Obs(n, 2);       
        Vd = V(:,dest - lastIndexNetworkState);    
        lnPn = 0;
        for i = 1: Op.n
             Gradient(n,i) = 0;
             gradVd(:,i) =  gradV(i).value (:,dest - lastIndexNetworkState);
        end
        sumInstU = 0;
        sumInstX = zeros(1,Op.n);        
        path = Obs(n,:);
        lpath = size(find(path),2);
        % Compute regular attributes
        for i = 2:lpath - 1
            sumInstU = sumInstU + (Ufull(path(i),path(i+1)) + Vd(min(path(i+1),lastIndexNetworkState + 1)) - Vd(path(i)))/Scale(path(i));
            for j = 1:Op.n
                sumInstX(j) = sumInstX(j) + (Atts(j).Value(path(i),path(i+1)) + gradVd(min(path(i+1),lastIndexNetworkState + 1),j) - gradVd(path(i),j))/Scale(path(i));
            end
        end
        Gradient(n,:) = Gradient(n,:) + sumInstX;
        lnPn = lnPn + sumInstU ;  
        LL =  LL + (lnPn - LL)/n;
        grad = grad + (Gradient(n,:) - grad)/n;
        Gradient(n,:) = - Gradient(n,:);
    end
    LL = -1 * LL; % IN ORDER TO HAVE A MIN PROBLEM
    grad =  - grad';
end

