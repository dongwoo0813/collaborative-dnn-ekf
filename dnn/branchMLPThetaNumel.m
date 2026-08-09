function [nTheta, arch] = branchMLPThetaNumel(cfg)
%{
File:
    dnn/branchMLPThetaNumel.m

Purpose:
    Return the number of trainable parameters in one configurable MLP branch.

This is a thin wrapper around branchMLPArchitecture so other code can keep
calling branchMLPThetaNumel(cfg).
%}

arch = branchMLPArchitecture(cfg);
nTheta = arch.nTheta;

end