function state = initKickbackMomentum(globalParams)
    flatParams = nn.ravel(globalParams);
    state = struct(
        "momentum", zeros(size(flatParams)),
        "currentParams", flatParams,
        "prevParams", flatParams,
        "mu1", 0.5,
        "mu2", 0.1
    );
endfunction
