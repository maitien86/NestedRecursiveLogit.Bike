function [f g] = LL(x)
    global Op;
    Op.x = x;
    [f g] = getLL_nested();
    %PrintOut(Op);
    Op.nFev  = Op.nFev + 1;
    Op.k = Op.nFev;
end