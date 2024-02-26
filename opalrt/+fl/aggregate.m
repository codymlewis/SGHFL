function [updates, aggParams] = aggregate(clientParams, fcn)
    cParams = fl.flattenClientParams(clientParams);
    aggParams = fcn(cParams)';
    updates = nn.unravel(aggParams, clientParams(1));
endfunction
