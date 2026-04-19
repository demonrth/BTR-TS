clear; clc; close all;
addpath(genpath('lib'));
addpath(genpath('data'));

missing_ways  = {'Random','Non-random1','Non-random2'}; 
missing_rates = [0.1, 0.3, 0.5, 0.7];%

datasets = {
    struct('name', 'Guangzhou', 'file', fullfile('data', 'guangzhou.mat'), 'varname', 'GuangZhou', 'layout', 'sensor_time_day')
    struct('name', 'PeMSD7M', 'file', fullfile('data', 'pemsD7M.mat'), 'varname', 'pemsT', 'layout', 'sensor_time_day')
    struct('name', 'PeMSD8',  'file', fullfile('data', 'pemsD8.mat'),  'varname', 'pemsT', 'layout', 'sensor_day_time')
};

fixed_params.R1    = 6;
fixed_params.R2    = 2;
fixed_params.R     = 17;
fixed_params.alpha = 0.5;
fixed_params.beta  = 1;
fixed_params.rho   = 0.5;
fixed_params.lam_t = 2;
fixed_params.tol   = 1e-4;
fixed_params.maxit = 150;

result_rows = {};

fprintf('Fixed parameters: [R1,R2,R]=[%d,%d,%d], Alpha=%g, Beta=%g, Rho=%.2f, LamT=%.3g\n', ...
    fixed_params.R1, fixed_params.R2, fixed_params.R, fixed_params.alpha, ...
    fixed_params.beta, fixed_params.rho, fixed_params.lam_t);

for ds_i = 1:length(datasets)
    ds = datasets{ds_i};
    [dense_tensor, original_size] = load_dataset_tensor(ds); 
    [n1, n2, n3] = size(dense_tensor); % canonical: [sensor, time, day]

    fprintf('\nDataset: %s\n', ds.name);

    for mw_i = 1:length(missing_ways)
        cur_way = missing_ways{mw_i};

        for mr_i = 1:length(missing_rates)
            cur_rate = missing_rates(mr_i);
            fprintf('Start running | Dataset: %s | Pattern: %s | Missing rate: %.2f\n', ds.name, cur_way, cur_rate);

            % --- A: Generate a mask under the canonical semantics ---
            rng(100);
            Pomega = generate_missing_mask([n1, n2, n3], cur_way, cur_rate);

            % --- B: Dimension mapping adaptation canonical [sensor,time,day] -> [sensor,day,t1,t2] ---
            [X_4D, mask_4D, target_shape] = map_tensor_to_4d(dense_tensor, Pomega);
            Omega = find(mask_4D == 1);
            F = zeros(target_shape);
            F(Omega) = X_4D(Omega);

            % Test set index
            test_idx = find(dense_tensor ~= 0 & Pomega == 0);
            opts = [];
            opts.tol      = fixed_params.tol;
            opts.maxit    = fixed_params.maxit;
            opts.allR     = [fixed_params.R1, fixed_params.R2, fixed_params.R];
            opts.alpha    = fixed_params.alpha;
            opts.beta1    = fixed_params.beta;
            opts.beta2    = fixed_params.beta;
            opts.rho      = fixed_params.rho;
            opts.lambda_t = fixed_params.lam_t;

            tStart = tic;
            try
                evalc('[Re_image, res, Out] = BTR_TS(F, Omega, opts);');
                runtime_sec = toc(tStart);
            catch ME                
                fprintf('FAILED (%)\n', ME.message);
                runtime_sec = NaN;
                result_rows(end+1, :) = { ...
                    ds.name, cur_way, cur_rate, ...
                    fixed_params.R1, fixed_params.R2, fixed_params.R, ...
                    fixed_params.alpha, fixed_params.beta, fixed_params.rho, fixed_params.lam_t, ...
                    inf, inf, runtime_sec};
                continue;
            end

            Recovered_X = map_tensor_back_to_3d(Re_image, target_shape, [n1, n2, n3]);

            if ~isempty(test_idx)
                truth = dense_tensor(test_idx);
                diff_val = truth - Recovered_X(test_idx);
                current_rmse = sqrt(mean(diff_val.^2));
                current_mae  = mean(abs(diff_val));
            else
                current_rmse = inf;
                current_mae  = inf;
            end

            fprintf('Result: RMSE = %.4f | MAE = %.4f | Time = %.4fs\n', current_rmse, current_mae, runtime_sec);

            result_rows(end+1, :) = { ...
                ds.name, cur_way, cur_rate, ...
                fixed_params.R1, fixed_params.R2, fixed_params.R, ...
                fixed_params.alpha, fixed_params.beta, fixed_params.rho, fixed_params.lam_t, ...
                current_rmse, current_mae, runtime_sec};
        end
    end
end

results_table = cell2table(result_rows, 'VariableNames', ...
    {'Dataset', 'Mode', 'Rate', 'R1', 'R2', 'R', 'Alpha', 'Beta', 'Rho', 'Lambda_T', 'RMSE', 'MAE', 'RuntimeSec'});
disp(' ');
disp('==================== Experiment completed; summary of results ====================');
disp(results_table);

try
    writetable(results_table, 'results.csv');
    fprintf('The results have been saved to results.csv\n');
catch ME
    fprintf('Failed to write CSV: %s\n', ME.message);
end

%% ======  helper functions ======
function [dense_tensor, original_size] = load_dataset_tensor(ds)
    S = load(ds.file);
    dense_tensor = double(S.(ds.varname));
    original_size = size(dense_tensor);

    switch ds.layout
        case 'sensor_time_day'
            % already [sensor, time, day]
        case 'sensor_day_time'
            dense_tensor = permute(dense_tensor, [1, 3, 2]);
    end
end

function Pomega = generate_missing_mask(sz, cur_way, cur_rate)
    n1 = sz(1);
    n2 = sz(2);
    n3 = sz(3); % [sensor, time, day]

    switch cur_way
        case 'Random'
            Pomega = round(rand(n1, n2, n3) + 0.5 - cur_rate);

        case 'Non-random1'
            % All sensors share the same (time, day) Missing pattern
            A = round(rand(n3, n2) + 0.5 - cur_rate);
            B = kron(A, ones(n1, 1));
            Pomega = reshape(B, [n1, n2, n3]);

        case 'Non-random2'
            %  certain (sensor, time) Missing at the same time on all days
            A = round(rand(n1, n2) + 0.5 - cur_rate);
            Pomega = repmat(A, [1, 1, n3]);
    end
end

function [X_4D, mask_4D, target_shape] = map_tensor_to_4d(dense_tensor, Pomega)
    [n1, n2, n3] = size(dense_tensor); % [sensor, time, day]

    tensor_perm = permute(dense_tensor, [1, 3, 2]); % [sensor, day, time]
    mask_perm   = permute(Pomega,      [1, 3, 2]);

    if n2 == 144
        t_dim1 = 12;
        t_dim2 = 12;    % Guangzhou
    elseif n2 == 288
        t_dim1 = 24;
        t_dim2 = 12;   % PeMSD7M / PeMSD8
    end

    target_shape = [n1, n3, t_dim1, t_dim2];
    X_4D    = reshape(tensor_perm, target_shape);
    mask_4D = reshape(mask_perm, target_shape);
end

function Recovered_X = map_tensor_back_to_3d(Re_image, target_shape, original_shape)
    n1 = original_shape(1);
    n2 = original_shape(2);
    n3 = original_shape(3);
    Output_4D   = reshape(Re_image, target_shape);
    Output_perm = reshape(Output_4D, [n1, n3, n2]);
    Recovered_X = permute(Output_perm, [1, 3, 2]);
end
