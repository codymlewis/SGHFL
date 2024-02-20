function [mae, r2] = test()
    nclients = 10;
    rounds = 10;
    nsamples = 1000;
    pctTrain = 0.7;
    X = -pi + (2 * pi) * rand(nsamples, 1);
    Y = sin(X);
    nTrainSamples = round(rows(X) * pctTrain);
    trainX = X(1:nTrainSamples, :);
    trainY = Y(1:nTrainSamples, :);
    testX = X(nTrainSamples:rows(X), :);
    testY = Y(nTrainSamples:rows(Y), :);
    idxs = randperm(rows(trainX))';
    nsamplesPerClient = round(nTrainSamples / nclients);
    clients = repmat(struct(), [nclients, 1]);
    for c = 1:nclients
        i = (c - 1) * nsamplesPerClient + 1;
        idx = idxs(i:min(i + nsamplesPerClient - 1, rows(trainX)));
        clients(c).X = trainX(idx);
        clients(c).Y = trainY(idx);
        clients(c).params = nn.initModel(columns(trainX), columns(trainY));
        clients(c).state = nn.initAdam(clients(c).params);
    endfor

    globalParams = nn.initModel(columns(trainX), columns(trainY));
    globalState = fl.initFedavg(globalParams);
    for r = 1:rounds
        lossValues = zeros([nclients, 1]);
        clientGrads = repmat(struct(), [nclients, 1]);
        for c = 1:nclients
            [lossValues(c), clients(c).params, clients(c).state, clientGrads(c)] = nn.train(
                globalParams, clients(c).state, clients(c).X, clients(c).Y, epochs=1
            );
        endfor
        [globalUpdates, globalState] = fl.applyFedavg(globalState, clientGrads);
        globalParams = nn.applyUpdates(globalParams, globalUpdates);
        disp(sprintf("Average training loss at round %d: %f", r, mean(lossValues)));
    endfor

    lossValue = nn.loss(globalParams, testX, testY);
    preds = nn.applyModel(globalParams, testX);
    mae = metrics.meanAbsoluteError(testY, preds);
    r2 = metrics.r2score(testY, preds);
    disp(sprintf("MAE: %.5f, r2 score: %.5f", mae, r2));
endfunction
