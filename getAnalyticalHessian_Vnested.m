% Compute the loglikelohood value and its gradient.
% Based on V - the mus are fixed
%%
function [LL, grad, Hessian, Hs] = getAnalyticalHessian_Vnested()
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

    %% ----------------------------------------------------
    [lastIndexNetworkState, maxDest] = size(incidenceFull);   
    Mfull = getM(Op.x, isLinkSizeInclusive);
    MregularNetwork = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Ufull = getU(Op.x, isLinkSizeInclusive);
    UregularNetwork = Ufull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    % Set LL value
    LL = 0;
    grad = zeros(1, Op.n);
    % Initialize
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
    V = (bsxfun(@times,mu,V));
    
    %% Get gradient    
    gradExpV = objArray(Op.n);
    gradV = objArray(Op.n);
    MI = sparse(M); 
    MI(find(M)) = 1;
    a = mu;
    k = 1 ./ mu;
    kf = k(1:lastIndexNetworkState,1);
    phi = sparse((k * a') .* MI);    
    e = ones(size(M,1),1);
    %% Get Gradient of M
    gradM = objArray(Op.n);   
    for i = 1: Op.n
        H = M .* (AttLc(i).Value);
        H = bsxfun(@times,k,H);
        gradM(i).value = H;
    end
    %% Compute gradient of t
    gradT = objArray(Op.n);   
    for i = 1:Op.n
        H =  Atts(i).Value(:,lastIndexNetworkState+1 : maxDest) .* Mfull(:,lastIndexNetworkState+1 : maxDest);
        H = bsxfun(@times,kf,H);
        H(lastIndexNetworkState+1,:) = sparse(zeros(1,maxDest - lastIndexNetworkState));     
        gradT(i).value = H;
    end
    %% Get Gradient of V
    for i = 1:Op.n
        gradVi = zeros(size(B));
        gradt = gradT(i).value;
        gradt = gradt(:,1:size(B,2)) .* bsxfun(@times,k,Z);
        for d = 1: size(B,2)
            % Compute P, N
            Zd = sparse(bsxfun(@times,Z(:,d)',MI));
            X = MI;
            X(find(MI)) =  Zd(find(MI)) .^ (phi(find(MI)));
            X = bsxfun(@ldivide,Z(:,d),X);
            H = M .* X;
            K =  bsxfun(@times, a ,(gradM(i).value .* X));
            gradVi(:,d) = (speye(size(M)) - H)\( K * e + gradt(:,d));  
        end
        gradV(i).value = gradVi;  
    end
    
    %% Compute gradient of Z    
    for i = 1:Op.n
        X = gradV(i).value .* Z;
        gradExpV(i).value = sparse(bsxfun(@times,k,X));
    end
    
    %% Evaluate hessian
    hessianM = objMatrix(Op.n, Op.n);
    hessianB = objMatrix(Op.n, Op.n);
    hessianV = objMatrix(Op.n, Op.n);
    Hessian = zeros(Op.n);
    H = zeros(Op.n);
    Hs = objArray(nbobs);
    %% Get second order derivative of M and B
    for i = 1: Op.n
        for j = 1: Op.n
            u = M .* AttLc(i).Value .* AttLc(j).Value;
            hessianM(i,j).value = bsxfun(@times,k .* k, u);
            u = Atts(i).Value(:,lastIndexNetworkState+1 : maxDest) .*  Atts(j).Value(:,lastIndexNetworkState+1 : maxDest) .* Mfull(:,lastIndexNetworkState+1 : maxDest);
            u = bsxfun(@times,kf .* kf,u);
            u(lastIndexNetworkState+1,:) = sparse(zeros(1,maxDest - lastIndexNetworkState));        
            hessianB(i,j).value = sparse(u);                        
        end
    end
    
    %% Get Hessian V
    for i = 1: Op.n
        for j = 1: Op.n
            Hes = zeros(size(B));     
            for d = 1:size(B,2)
                d
                % X = z^phi/z
                Zd = sparse(bsxfun(@times,Z(:,d)',MI));
                X = MI;
                X(find(MI)) =  Zd(find(MI)) .^ (phi(find(MI)));
                X = bsxfun(@ldivide,Z(:,d),X);
                % compute H
                H = M .* X;
                % Compute U
                U1 = bsxfun(@times,a,hessianM(i,j).value .* X);
                T = (gradM(i).value .* X);
                U2e = T * gradV(j).value(:,d);
                U3  = - bsxfun(@times,gradV(j).value(:,d) ,T);
                U4e =  (gradM(j).value .* X) * gradV(i).value(:,d);
                U5e = bsxfun(@times, k, M .* X) * (gradV(i).value(:,d) .* gradV(j).value(:,d));
                U6e = -bsxfun(@times, k .*  gradV(j).value(:,d), (M .* X)) * gradV(i).value(:,d);
                Ld  = hessianB(i,j).value(:,d) .* Z(:,d) .* k + gradT(i).value(:,d) .* gradV(j).value(:,d) .* Z(:,d) .* k .* k;               
                Hes(:,d) = (speye(size(M)) - H)\((U1+U3)*e + Ld + U2e + U4e + U5e + U6e);  
            end
            hessianV(i,j).value = Hes;
        end
    end
        
    %% Compute the LL and gradient.
    gradVd = zeros(size(Z,1),Op.n);
    hesVd = objMatrix(Op.n, Op.n);
    for n = 1:nbobs    
        dest = Obs(n, 1);
        orig = Obs(n, 2);       
        Vd = V(:,dest - lastIndexNetworkState);    
        lnPn = 0;
        for i = 1: Op.n
             Gradient(n,i) = 0;
             gradVd(:,i) =  gradV(i).value (:,dest - lastIndexNetworkState);
             for j = 1:Op.n
                 hesVd(i,j).value = hessianV(i,j).value(:,dest - lastIndexNetworkState);
             end
        end
        sumInstU = 0;
        sumInstX = zeros(1,Op.n);       
        sumInstH = zeros(Op.n,Op.n);
        path = Obs(n,:);
        lpath = size(find(path),2);
        % Compute regular attributes
        for i = 2:lpath - 1
            sumInstU = sumInstU + (Ufull(path(i),path(i+1)) + Vd(min(path(i+1),lastIndexNetworkState + 1)) - Vd(path(i)))/mu(path(i));
            for j = 1:Op.n
                sumInstX(j) = sumInstX(j) + (Atts(j).Value(path(i),path(i+1)) + gradVd(min(path(i+1),lastIndexNetworkState + 1),j) - gradVd(path(i),j))/mu(path(i));
            end
            for x = 1: Op.n
                for y = 1: Op.n
                    sumInstH(x,y) = sumInstH(x,y) + (hesVd(x,y).value(min(path(i+1),lastIndexNetworkState + 1)) - hesVd(x,y).value(path(i)))/ mu(path(i));
                end
            end
        end
        Hs(n).value = sumInstH;
        Hessian = Hessian + (sumInstH - Hessian)/n;
        Gradient(n,:) = Gradient(n,:) + sumInstX;
        lnPn = lnPn + sumInstU ;  
        LL =  LL + (lnPn - LL)/n;
        grad = grad + (Gradient(n,:) - grad)/n;
        Gradient(n,:) = - Gradient(n,:);
    end
    LL = -1 * LL; % IN ORDER TO HAVE A MIN PROBLEM
    grad =  - grad';
end

