function params = initModel(inputSize, outputSize)
    params = struct(
        "linear1", nn.initLayer(inputSize, 16),
        "linear2", nn.initLayer(16, 6),
        "linear3", nn.initLayer(6, outputSize)
    );
endfunction
