function state = initAdam(params, beta1=0.9, beta2=0.999, epsilon=1e-8)
    state = struct('beta1', beta1, 'beta2', beta2, 'epsilon', epsilon, 't', 0, 'm', struct(), 'v', struct());
    paramsFields = fieldnames(params);
    for i = 1:numel(paramsFields)
        state.m.(paramsFields{i}) = struct(
            'W', zeros(size(params.(paramsFields{i}).W)),
            'b', zeros(size(params.(paramsFields{i}).b))
        );
        state.v.(paramsFields{i}) = struct(
            'W', zeros(size(params.(paramsFields{i}).W)),
            'b', zeros(size(params.(paramsFields{i}).b))
        );
    endfor
endfunction
