function params = applyUpdates(params, updates)
    updatesFields = fieldnames(updates);
    for i = 1:numel(updatesFields)
        params.(updatesFields{i}).W -= updates.(updatesFields{i}).W;
        params.(updatesFields{i}).b -= updates.(updatesFields{i}).b;
    endfor
endfunction
