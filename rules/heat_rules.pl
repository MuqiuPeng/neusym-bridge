% Phase 2 output: predicates from intervention experiments
% temperature_dominant(Node)  — primary temperature mode (effect=0.1262)
% temperature_global(Node)    — global temperature level (effect=0.0509)
% temperature_spatial(Node)   — spatial structure mode (effect=0.0200)

% Three modes simultaneously active → heat concentration state
heat_concentration(N) :- temperature_dominant(N), temperature_global(N), temperature_spatial(N).

% Heat concentration → structural risk
structural_risk(N) :- heat_concentration(N).
