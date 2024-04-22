function clientState = clientInit(trainX, trainY)
    params = nn.initModel(columns(trainX), columns(trainY));
    clientState = struct(
        'X', trainX,
        'Y', trainY,
        'params', params,
        'state', nn.initAdam(params)
    );
endfunction
