function state = initFedProx(globalParams)
    flatParams = nn.ravel(globalParams);
    state = struct(
        "currentParams", flatParams,
        "prevParams", flatParams,
        "iteration", 1,
        "mu", 0.0001
    );
endfunction
