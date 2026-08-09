function out = run_all_step04_smoke_tests(mode)
%{
Function:
    run_all_step04_smoke_tests

Purpose:
    Run the existing Step 04 sanity/smoke checks as one test suite.

What this checks:
    1. Step 04 GS-composite architecture checks
    2. Step 04b nonlocal covariance injection checks
    3. Step 04c block covariance prediction checks

Modes:
    "quick":
        Fast checks only.
        - run_step04_sanity_checks("quick")
        - check_step04c_block_covariance_prediction("quick")

    "full":
        Full checks.
        - run_step04_sanity_checks("all")
        - check_step04b_nonlocal_covariance_injection()
        - check_step04c_block_covariance_prediction("full")

Inputs:
    mode - "quick" or "full"
           Default: "quick"

Outputs:
    out - Struct containing pass/fail information and each check output.

How to run:
    Quick smoke test:
        out = run_all_step04_smoke_tests();

    Full smoke test:
        out = run_all_step04_smoke_tests("full");

Notes:
    This runner does not introduce new validation logic.
    It only organizes existing check files so that debugging is easier.
%}

    if nargin < 1 || isempty(mode)
        mode = "quick";
    end

    mode = string(mode);

    if mode ~= "quick" && mode ~= "full"
        error("Unsupported mode: %s. Use 'quick' or 'full'.", mode);
    end

    % ---------------------------------------------------------------------
    % Path setup
    % ---------------------------------------------------------------------
    thisFile = mfilename("fullpath");
    thisDir = fileparts(thisFile);
    projectRoot = fileparts(thisDir);

    addpath(genpath(projectRoot));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("All Step 04 smoke tests\n");
    fprintf("Mode: %s\n", mode);
    fprintf("Project root: %s\n", projectRoot);
    fprintf("============================================================\n");

    out = struct();
    out.mode = mode;
    out.projectRoot = projectRoot;
    out.results = struct();
    out.summary = table();

    testNames = strings(0,1);
    testPassed = false(0,1);
    testMessage = strings(0,1);

    % ---------------------------------------------------------------------
    % Test 1: Step 04 GS-composite architecture
    % ---------------------------------------------------------------------
    if mode == "quick"
        [passed, msg, result] = runOneSmokeTest( ...
            "Step04_GS_architecture_quick", ...
            @() run_step04_sanity_checks("quick"));
    else
        [passed, msg, result] = runOneSmokeTest( ...
            "Step04_GS_architecture_full", ...
            @() run_step04_sanity_checks("all"));
    end

    out.results.step04GSArchitecture = result;
    testNames(end+1,1) = "Step04_GS_architecture"; 
    testPassed(end+1,1) = passed; 
    testMessage(end+1,1) = msg; 

    % ---------------------------------------------------------------------
    % Test 2: Step 04b Qnonlocal injection
    % Skip in quick mode because it runs multiple simulations.
    % ---------------------------------------------------------------------
    if mode == "full"
        [passed, msg, result] = runOneSmokeTest( ...
            "Step04b_Qnonlocal_injection", ...
            @() check_step04b_nonlocal_covariance_injection());

        out.results.step04bQnonlocal = result;
        testNames(end+1,1) = "Step04b_Qnonlocal_injection"; 
        testPassed(end+1,1) = passed; 
        testMessage(end+1,1) = msg; 
    end

    % ---------------------------------------------------------------------
    % Test 3: Step 04c block covariance prediction
    % ---------------------------------------------------------------------
    if mode == "quick"
        [passed, msg, result] = runOneSmokeTest( ...
            "Step04c_block_covariance_quick", ...
            @() check_step04c_block_covariance_prediction("quick"));
    else
        [passed, msg, result] = runOneSmokeTest( ...
            "Step04c_block_covariance_full", ...
            @() check_step04c_block_covariance_prediction("full"));
    end

    out.results.step04cBlockCovariance = result;
    testNames(end+1,1) = "Step04c_block_covariance"; 
    testPassed(end+1,1) = passed; 
    testMessage(end+1,1) = msg; 


    % ---------------------------------------------------------------------
    % Test 4: Step 05 event-triggered GS upload
    % Full mode only because it runs frequent and event-trigger simulations.
    % ---------------------------------------------------------------------
    if mode == "full"
        [passed, msg, result] = runOneSmokeTest( ...
            "Step05_event_triggered_upload", ...
            @() check_step05_event_triggered_upload());

        out.results.step05EventTriggeredUpload = result;
        testNames(end+1,1) = "Step05_event_triggered_upload"; 
        testPassed(end+1,1) = passed; 
        testMessage(end+1,1) = msg; 
    end


    % ---------------------------------------------------------------------
    % Summary
    % ---------------------------------------------------------------------
    out.summary = table( ...
        testNames, ...
        testPassed, ...
        testMessage, ...
        'VariableNames', {'testName', 'passed', 'message'});

    out.allPassed = all(testPassed);
    out.numPassed = nnz(testPassed);
    out.numFailed = numel(testPassed) - out.numPassed;

    fprintf("\n============================================================\n");
    fprintf("All Step 04 smoke-test summary\n");
    fprintf("============================================================\n");
    disp(out.summary);

    fprintf("Passed: %d / %d\n", out.numPassed, numel(testPassed));

    if ~out.allPassed
        error("%d Step 04 smoke test(s) failed. See summary above.", out.numFailed);
    end

    fprintf("\nAll requested Step 04 smoke tests passed.\n");

end

function [passed, msg, result] = runOneSmokeTest(testName, testFcn)
% Run one smoke test and convert errors into pass/fail summary rows.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Running smoke test: %s\n", testName);
    fprintf("------------------------------------------------------------\n");

    try
        result = testFcn();
        passed = true;
        msg = "passed";
        fprintf("PASS: %s\n", testName);
    catch ME
        result = struct();
        passed = false;
        msg = string(ME.message);
        fprintf("FAIL: %s\n", testName);
        fprintf("Reason: %s\n", ME.message);
    end

end