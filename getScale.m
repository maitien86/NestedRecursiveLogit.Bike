%% Compute ET and LS for the Scales
function [] = getScale()
    global incidenceFull;  
    global Scale;
    global Atts;
    %% Get link length
    
    LLength =  Atts(1).value;
    maxstates = size(LLength,1);
    ET = zeros(size(LLength,2),1);
    I = find(LLength);
    [nbnonzero] = size(I,1);
    for i = 1:nbnonzero
        [k,a] = ind2sub(size(LLength), I(i));
        ET(a,1) = LLength(k,a);     
    end
    ET = ET(1:maxstates+1);
    ET = ET / max(ET);
    %ONE = ones(maxstates+1,1); 
    %% Compute flow
%     lastIndexNetworkState = size(incidenceFull,1);
%     %beta = [-1,0,0,0]';
%     beta = [-5.2570;-4.6702;-5.0949;-1.5374;-5.1486]';
%     [ok, flow] = getFlow(beta);
%     if ok == false
%         fprintf(' Beta is wrong, try other parameters \n');
%         flow = ones(lastIndexNetworkState,1);
%     end
%     flow = flow(1: maxstates + 1);
%     flow = flow / max(flow);
    
    %% Link length
    MI = sparse(LLength); 
    MI(find(LLength)) = 1;
    AT = sum(LLength,2) ./ sum(MI,2);
    AT(maxstates+1,1) = 0;
    AT = AT / max(AT);
    %% Number of outgoing links
    Nb = sum(incidenceFull,2);
    Nb = [Nb;0];
    %% Set Scales
    Scale = zeros(maxstates + 1,1);
    %Scale(:,1) = ET;
    %Scale(:,1) = flow;
    Scale(:,1) = Nb;
    %Scale(:,2) = ONE;
end