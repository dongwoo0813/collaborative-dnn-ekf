function neighbors = getP2PNeighbors(i,Nw)
%GETP2PNEIGHBORS Two-neighbor ring graph used by Step 10-B1.

if Nw <= 1
    neighbors = zeros(1,0);
    return;
elseif Nw == 2
    neighbors = 3-i;
else
    neighbors = [mod(i-2,Nw)+1, mod(i,Nw)+1];
end
neighbors = unique(neighbors(neighbors ~= i));
end
