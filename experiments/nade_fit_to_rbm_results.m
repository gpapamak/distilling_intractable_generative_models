% Shows the results of fitting a nade to the rbm.
%
% George Papamakarios, Jul 2015

clear;
close all;

% folder where to read results from
outdir = fullfile('outdir');

% number of hidden units in nade
num_hidden = [1000, 750, 500, 250];

% platform where to run, can be 'cpu' or 'gpu'
platform = 'cpu';

% parameters for the plots
fontsize = 18;
linewidth = 3;

% load rbm
load(fullfile(outdir, 'rbm_CD25_500.mat'), 'rbm');
rbm.changePlatform(platform);

%% plot training progress

% colours for the plots
col = [1, 0, 0; 0, 1, 0; 0, 0, 1; 0 0 0];

% kl divergence
figure; hold on;
for i = 1:numel(num_hidden)
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', num_hidden(i))));
    xx = batch_size * (0 : monitor_every : maxiter) / 1000;
    plot(xx, progress.tst, 'Color', col(i,:), 'LineWidth', linewidth);
end
xlabel('Thousands of RBM samples', 'FontSize', fontsize);
ylabel('Average test log probability', 'FontSize', fontsize);
legend('1000 hidden units', '750 hidden units', '500 hidden units', '250 hidden units', 'Location', 'SouthEast');
title('KL divergence', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);

% square error
figure; hold on;
for i = 1:numel(num_hidden)
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', num_hidden(i))));
    xx = batch_size * (0 : monitor_every : maxiter) / 1000;
    plot(xx, progress.tst, 'Color', col(i,:), 'LineWidth', linewidth);
end
xlabel('Thousands of RBM samples', 'FontSize', fontsize);
ylabel('Average test log probability', 'FontSize', fontsize);
legend('1000 hidden units', '750 hidden units', '500 hidden units', '250 hidden units', 'Location', 'SouthEast');
title('Square error', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);

%% calculate log probability on test set

% load mnist; Salakhutdinov and Murray's binarization
load(fullfile(outdir, 'randomly_binarized_mnist.mat'), 'x_tst');
num_tst = size(x_tst, 2);
bufsize = 500;

% kl divergence
fprintf('** kl divergence ** \n');
for i = 1:numel(num_hidden)
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', num_hidden(i))));
    
    L = zeros(1, num_tst);
    for j = 1:num_tst/bufsize
        idx = (j-1) * bufsize + 1 : j * bufsize;
        L(idx) = nade.eval(x_tst(:,idx));
    end
    fprintf('%d hiddens = %.2f +/- %.2f \n', num_hidden(i), mean(L), 3*std(L) / sqrt(num_tst));
end
fprintf('\n');

% square error
fprintf('** square error ** \n');
for i = 1:numel(num_hidden)
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', num_hidden(i))));
    
    L = zeros(1, num_tst);
    for j = 1:num_tst/bufsize
        idx = (j-1) * bufsize + 1 : j * bufsize;
        L(idx) = nade.eval(x_tst(:,idx));
    end
    fprintf('%d hiddens = %.2f +/- %.2f \n', num_hidden(i), mean(L), 3*std(L) / sqrt(num_tst));
end
fprintf('\n');

%% plot log probability on test set

logZ = 451.28;      % this is the logZ estimate by Salakhutdinov and Murray
x_tst = x_tst(:, 1:500);

% evaluate rbm
L_rbm = rbm.eval(x_tst) - logZ;
[L_rbm, idx] = sort(L_rbm, 'descend');

% kl divergence
for i = num_hidden
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    L_nade = nade.eval(x_tst);
    L_nade = L_nade(idx);
    figure; hold on;
    plot(L_nade, 'r', 'LineWidth', linewidth);
    plot(L_rbm, 'b', 'LineWidth', linewidth);
    title(sprintf('KL divergence, %d hidden units', i), 'FontSize', fontsize);
    xlabel('Test image index', 'FontSize', fontsize);
    ylabel('Log probability', 'FontSize', fontsize);
    legend('NADE', 'RBM', 'Location', 'SouthWest');
    set(gca, 'FontSize', fontsize);
end

% square error
for i = num_hidden
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', i)), 'nade');
    L_nade = nade.eval(x_tst);
    L_nade = L_nade(idx);
    figure; hold on;
    plot(L_nade, 'r', 'LineWidth', linewidth);
    plot(L_rbm, 'b', 'LineWidth', linewidth);
    title(sprintf('Square error, %d hidden units', i), 'FontSize', fontsize);
    xlabel('Test image index', 'FontSize', fontsize);
    ylabel('Log probability', 'FontSize', fontsize);
    legend('NADE', 'RBM', 'Location', 'SouthWest');
    set(gca, 'FontSize', fontsize);
end

%% calculate kl divergences

num_std = 3;

% kl divergence
fprintf('** kl divergence ** \n');
for i = num_hidden
    fprintf('%d hidden units \n', i);
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)));
    nade.changePlatform(platform);
    [diff, stdev] = rbm.difference(nade, 'kl');
    fprintf('    KL(rbm||nade) = %.2f +/- %.2f \n', diff, num_std*stdev);
    [diff, stdev] = nade.difference(rbm, 'kl');
    fprintf('    KL(nade||rbm) = %.2f +/- %.2f \n', diff, num_std*stdev);
end
fprintf('\n');

% square error
fprintf('** square error ** \n');
for i = num_hidden
    fprintf('%d hidden units \n', i);
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', i)));
    nade.changePlatform(platform);
    [diff, stdev] = rbm.difference(nade, 'kl');
    fprintf('    KL(rbm||nade) = %.2f +/- %.2f \n', diff, num_std*stdev);
    [diff, stdev] = nade.difference(rbm, 'kl');
    fprintf('    KL(nade||rbm) = %.2f +/- %.2f \n', diff, num_std*stdev);
end
fprintf('\n');
