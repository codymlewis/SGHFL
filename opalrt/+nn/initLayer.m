function params = initLayer(inputSize, outputSize)
    l = sqrt(6 / (inputSize + outputSize));
    W = -l + (2*l) .* rand(inputSize, outputSize);
    b = zeros(outputSize, 1);
    params = struct("W", W, "b", b);
endfunction
