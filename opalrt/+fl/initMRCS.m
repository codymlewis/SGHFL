function state = initMRCS(globalParams)
    flatParams = nn.ravel(globalParams);
    state = struct(
        "currentParams", flatParams,
        "momentum", zeros(size(flatParams)),
        "mu", 0.7
    );
endfunction
