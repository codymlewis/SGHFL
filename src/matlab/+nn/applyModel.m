function [output, layerOutput] = applyModel(params, X)
    nsamples = rows(X);
    output = zeros(nsamples, columns(params.linear3.W));
    for i = 1:nsamples
        o1 = nn.linear(params.linear1, X(i, :)');
        o1 = nn.relu(o1);
        o2 = nn.linear(params.linear2, o1);
        o2 = nn.relu(o2);
        o3 = nn.linear(params.linear3, o2);
        output(i,:) = o3;
    endfor
endfunction
