function tf = isP2PCommunication(cfg)
%ISP2PCOMMUNICATION Return true for the decentralized ring-cache backend.

tf = false;
if isfield(cfg,"communication") && isfield(cfg.communication,"architecture")
    tf = string(cfg.communication.architecture) == "p2p_ring";
elseif isfield(cfg,"gs") && isfield(cfg.gs,"architecture")
    tf = string(cfg.gs.architecture) == "p2p_ring";
end
end
