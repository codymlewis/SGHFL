function [lossValue, params, state, pseudoGradient] = train(startingParams, state, trainX, trainY, epochs=10, batchSize=128)
    params = startingParams;
    for e = 1:epochs
        idxs = randperm(rows(trainX))';
        for i = 1:batchSize:rows(trainX)
            idx = idxs(i:min(i + batchSize, rows(trainX)));
            grads = nn.backprop(params, trainX(idx, :), trainY(idx, :));
            [updates, state] = nn.applyAdam(state, grads);
            params = nn.applyUpdates(params, updates);
        endfor
        lossValue = nn.loss(params, trainX, trainY);
        pseudoGradient = nn.subtractParams(startingParams, params);
    endfor
endfunction
