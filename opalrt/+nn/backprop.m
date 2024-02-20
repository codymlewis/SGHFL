function grads = backprop(params, X, Y)
    ## Setup cache
    N = rows(X);
    paramsFields = fieldnames(params);
    layerOutputs = struct('linear0', X);
    for i = 1:numel(paramsFields)
        layerOutputs.(paramsFields{i}) = zeros([N, columns(params.(paramsFields{i}).W)]);
    endfor

    ## Forward pass
    layerOutputsFields = fieldnames(layerOutputs);
    for i = 1:N
        for j = 2:numel(layerOutputsFields)
            layerOutputs.(layerOutputsFields{j})(i, :) = nn.linear(
                params.(layerOutputsFields{j}), layerOutputs.(layerOutputsFields{j - 1})(i, :)'
            );
            if j < numel(layerOutputsFields)
                layerOutputs.(layerOutputsFields{j})(i, :) = nn.relu(layerOutputs.(layerOutputsFields{j})(i, :));
            endif
        endfor
    endfor

    ## Back-propagate the gradients
    grads = struct();
    delta = (1 / N) * (layerOutputs.linear3 - Y)';
    grads.(paramsFields{numel(paramsFields)}) = struct(
        'W', (delta * layerOutputs.(layerOutputsFields{numel(layerOutputsFields) - 1}))',
        'b', sum(delta, 2)
    );
    for i = numel(layerOutputsFields):-1:3
        delta = params.(layerOutputsFields{i}).W * delta;
        grads.(layerOutputsFields{i - 1}) = struct(
            'W', (delta * layerOutputs.(layerOutputsFields{i - 2}))',
            'b', sum(delta' .* (layerOutputs.(layerOutputsFields{i - 1}) > 0))'
        );
    endfor
endfunction
