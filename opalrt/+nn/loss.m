function value = loss(params, X, Y)
    predictions = nn.applyModel(params, X);
    value = mean((0.5 .* (predictions - Y).^2), 'all');
endfunction
