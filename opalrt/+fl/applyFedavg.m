function [updates, state] = applyFedavg(state, clientGrads)
    nclients = rows(clientGrads);
    cParams = fl.as2sa(clientGrads);
    cParamsFields = fieldnames(cParams);
    updates = struct();
    for i = 1:numel(cParamsFields)
        meanWeights = mean(cParams.(cParamsFields{i}).W, 1);
        if numel(size(meanWeights)) > 2
            meanWeights = reshape(meanWeights, size(meanWeights)(2:3));
        else
            meanWeights = meanWeights';
        endif
        updates.(cParamsFields{i}).W = meanWeights;
        updates.(cParamsFields{i}).b = reshape(
            mean(cParams.(cParamsFields{i}).b, 1),
            [columns(cParams.(cParamsFields{i}).b), 1]
        );
    endfor
endfunction
