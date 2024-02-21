function state = initFedProx(globalParams)
    state = struct(
        "currentParams", globalParams,
        "prevParams", globalParams,
        "iteration", 1
    );
endfunction
