%   Get Utility
%%
function Ufull = getU(x, isLS)
    global incidenceFull;
    global Atts;
    %global nParams_v;
    global Op;
    u = 0 * Atts(1).value;
    for i = 1:Op.m
        u = u + x(i) * Atts(i).value;
    end
    u = sparse(u);
    Ufull = incidenceFull .* u;
end


