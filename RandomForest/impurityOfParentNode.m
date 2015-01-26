function bits = impurityOfParentNode(labels)
% Compute cardinality of feature
vals = unique(labels);
cardinality = length(vals);
N = length(labels);
bits = 0;
for i = 1 : cardinality
    
    Ni = length(find(labels == vals(i)));
    p = Ni / N;
    bits = bits - (p * log2(p));
end
