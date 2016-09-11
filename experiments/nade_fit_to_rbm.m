% Trains a nade to mimic an rbm.
%
% George Papamakarios, Jul 2015

clear;
close all;

% the platform where to run the experiment
% can be 'cpu' or 'gpu'
% note that it requires the Parallel Computing Toolbox
platform = 'gpu';

% folder where to save results
outdir = fullfile('outdir');

% make the arrangements for the platform
switch platform
    case 'cpu'
        createArray = @(x) x;
    case 'gpu'
        createArray = @gpuArray;
    otherwise
        error('Unknown platform.');
end

% load mnist
% note that this is Salakhutdinov and Murray's binarization
% we need to use this because the rbm was trained on data binarized this way
load(fullfile(outdir, 'randomly_binarized_mnist.mat'), 'x_trn', 'x_tst');
x_trn = createArray(x_trn);
x_tst = createArray(x_tst(:, 1:500));

% load rbm
load(fullfile(outdir, 'rbm_CD25_500.mat'));
rbm.changePlatform(platform);

% train parameters
batch_size = 20;          % the minibatch size for training
maxiter = 30000;          % number of iterations to train for
monitor_every = 200;      % evaluate on test set every that many iterations
type = 'logistic';        % the type of hidden units in nade
num_hidden = 500;         % the number of hidden units in nade
loss_fun = {'max_likelihood', 'square_error'};    % loss function for training
burnin = 2000;            % burnin period for the gibbs sampler that samples from rbm
num_chains = 100 * batch_size;    % number of parallel gibbs chains

for i = 1:numel(loss_fun)
    
    % initialize rbm stream
    rbm.setGibbsState(double(rand(rbm.num_inputs, num_chains) > 0.5));
    stream = RbmStream(rbm, burnin);

    % train nade
    nade = Nade(rbm.num_inputs, num_hidden, type, platform, 1:rbm.num_inputs, 'single');
    progress = nade.train_stream(stream, 'loss', loss_fun{i}, 'minibatch', batch_size, 'maxiter', maxiter, 'monitor_every', monitor_every, 'x_tst', x_tst);
    
    % save nade and training progress
    outfile = sprintf('mimic_rbm_%d_%s_%s.mat', num_hidden, type, loss_fun{i});
    nade.changePlatform('cpu');
    save(fullfile(outdir, outfile), 'nade', 'progress', 'maxiter', 'batch_size', 'monitor_every', 'burnin', 'num_chains');
end
