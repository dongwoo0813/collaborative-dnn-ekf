function destination = mergeP2PRepositories(destination,source)
%MERGEP2PREPOSITORIES Merge the newest branch records from a peer library.
% Communication remains neighbor-to-neighbor, but each transmitted packet
% contains the sender's complete cache.  Version and source timestamp decide
% whether a received record is newer than the destination copy.

if ~isfield(destination,"branch") || ~isfield(source,"branch")
    error("mergeP2PRepositories:MissingBranchLibrary", ...
        "Both repositories must contain branch records.");
end
if numel(destination.branch) ~= numel(source.branch)
    error("mergeP2PRepositories:BranchCountMismatch", ...
        "Peer repositories must contain the same number of branches.");
end

for j = 1:numel(source.branch)
    incoming = source.branch(j);
    current = destination.branch(j);
    if string(incoming.status) ~= "valid"
        continue;
    end
    incomingVersion = double(incoming.version);
    currentVersion = double(current.version);
    incomingTime = double(incoming.lastUpdateTime);
    currentTime = double(current.lastUpdateTime);
    currentValid = string(current.status) == "valid";
    isNewer = ~currentValid || incomingVersion > currentVersion || ...
        (incomingVersion == currentVersion && incomingTime > currentTime);
    if isNewer
        destination.branch(j) = incoming;
    end
end
end
