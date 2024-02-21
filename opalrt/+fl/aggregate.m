function updates = aggregate(state, clientParams, fcn)
    cParams = fl.as2sa(clientParams);
    cParamsFields = fieldnames(cParams);
    updates = struct();
    for i = 1:numel(cParamsFields)
        aggWeights = fcn(cParams.(cParamsFields{i}).W, state, cParamsFields{i});
        if numel(size(aggWeights)) > 2
            aggWeights = reshape(aggWeights, size(aggWeights)(2:3));
        else
            aggWeights = aggWeights';
        endif
        updates.(cParamsFields{i}) = struct(
            'W', aggWeights,
            'b', reshape(
                fcn(cParams.(cParamsFields{i}).b, state, cParamsFields{i}), [columns(cParams.(cParamsFields{i}).b), 1]
            )
        );
    endfor
endfunction
