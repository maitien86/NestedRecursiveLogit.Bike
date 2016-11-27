% Compute the loglikelohood value and its gradient.
% Based on V -  Relaxing scales mu.
%%
function [LL, grad] = getLL_nested_DeC()

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
    global mu;
    global Scale;
    global isFixedMu;

    %% If Mu is fixed
    if isFixedMu == true
       [LL, grad] = getLL2_nested();
       return;
    end
        
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
    for i = 1:Op.m
        AttLc(i) =  Matrix2D(Atts(i).Value(1:lastIndexNetworkState,1:lastIndexNetworkState));
        AttLc(i).Value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        AttLc(i).Value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end
    for i = Op.m+1: Op.n
        AttLc(i) =  sparse(zeros(size(M)));
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
    B = B(:,1:1);
    [Z, expVokBool]   = getZ_DeC(M, B);
    if (expVokBool == 0)
            LL = OptimizeConstant.LL_ERROR_VALUE;
            grad = ones(Op.n,1);
            disp('The parameters not fesible')
            return; 
    end
    
    %% Compute V
    V = log(Z);
    V = (bsxfun(@times,mu,V));
    %% Compute gradient of Mu
    gradMu = zeros(N,Op.n);
    for i = Op.m+1 : Op.n
        gradMu(:,i) = mu .* Scale(:,i - Op.m);
    end
    %% Compute phi(a|k);    
    MI = sparse(M); 
    MI(find(M)) = 1;
    dPhi = objArray(Op.n - Op.m);
    a = mu;
    k = 1 ./ mu;
    kf = k(1:lastIndexNetworkState,1);
    for i = Op.m + 1: Op.n
       dPhi(i).value = sparse(bsxfun(@plus,-Scale(:,i - Op.m),Scale(:,i - Op.m)') .* MI) ; % dPhi(k,a) = Scale(a) - Scale(k).    
    end        
    phi = sparse((k * a') .* MI);    
    e = ones(size(M,1),1);
    %% Compute gradient of phi(.)
    gradPhi = objArray(Op.n);
    for i = 1: Op.n
        if i <= Op.m
           gradPhi(i).value = sparse(zeros(size(Z,1))); 
        else
           gradPhi(i).value = sparse(phi .* dPhi(i).value);
        end
    end    
    %% Compute gradient of V - respect to attributes parameters 
    gradExpV = objArray(Op.n);
    gradV = objArray(Op.n);
    for i = 1:Op.n
        gradVi = zeros(size(B)); 
        % Compute h^d      
        U1 = bsxfun(@times,kf, Atts(i).Value(:,lastIndexNetworkState+1 : maxDest));
        U2 = bsxfun(@times,kf .* kf .* gradMu(1:lastIndexNetworkState,i) , Ufull(:,lastIndexNetworkState+1 : maxDest)); 
        gradt = sparse((U1 - U2) .* Mfull(:,lastIndexNetworkState+1 : maxDest));
        gradt(lastIndexNetworkState+1,:) = sparse(zeros(1,maxDest - lastIndexNetworkState));        
        hd = gradt(:,1:size(B,2)) ./ Z;
        hd = bsxfun(@times,a, hd) + bsxfun(@times,gradMu(:,i),log(Z));        
        % Compute gradient of M
        U1 = bsxfun(@times,k, AttLc(i).Value);
        U2 = bsxfun(@times,k .* k .* gradMu(:,i) , U); 
        gradM = sparse(M .* (U1 - U2));
        % Compute gradient of V
        for d = 1: size(B,2)
            % Compute H, K
            Zd = sparse(bsxfun(@times,Z(:,d)',MI));
            X = MI;
            X(find(MI)) =  Zd(find(MI)) .^ (phi(find(MI)));
            X = bsxfun(@ldivide,Z(:,d),X);
            H = M .* X;
            U1 = bsxfun(@times, log(Z(:,d)'), H);
            U2 = bsxfun(@times, a, U1) .* gradPhi(i).value;
            U3 = bsxfun(@times, gradMu(:,i)', U1);
            S =  bsxfun(@times, a ,(gradM .* X)) + U2 - U3;         
            gradVi(:,d) = (speye(size(M)) - H)\( S * e + hd(:,d));  
        end
        gradV(i).value = gradVi;  
    end
    %% Compute gradient of Z    
    for i = 1:Op.n
        X = gradV(i).value - bsxfun(@times,gradMu(:,i),log(Z));
        X = X .* Z;
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
            sumInstU = sumInstU + (Ufull(path(i),path(i+1)) + Vd(min(path(i+1),lastIndexNetworkState + 1)) - Vd(path(i)))/mu(path(i));
            for j = 1:Op.n
                p1 =   (Atts(j).Value(path(i),path(i+1)) + gradVd(min(path(i+1),lastIndexNetworkState + 1),j) - gradVd(path(i),j))/mu(path(i));
                p2 =   (Ufull(path(i),path(i+1)) + Vd(min(path(i+1),lastIndexNetworkState + 1)) - Vd(path(i)))/(mu(path(i))^2) * gradMu(path(i),j) ;
                sumInstX(j) = sumInstX(j) + p1 - p2;
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

%%
