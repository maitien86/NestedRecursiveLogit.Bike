%   Get MUtility
%%
function Mfull = getM(x,kM,isLS)   

    global incidenceFull;
    global Atts;
    %global nParams_v;
    global Op;
    u = 0 * Atts(1).value;
    for i = 1:Op.m
        u = u + x(i) * Atts(i).value;
    end
    u = sparse(u);
    sizeU = size(u,1);
    kM = kM(1:sizeU,1:sizeU); 
    u = kM * u;
    expM = u ;
    expM(find(incidenceFull)) = exp(u(find(incidenceFull)));
    Mfull = incidenceFull .* expM;
end