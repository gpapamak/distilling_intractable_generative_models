% Generates and prints samples from binarized mnist, the rbm trained on it
% and the various nades that were trained to mimic it.
%
% George Papamakarios, Jul 2015

clear;
close all;

% folder to read results from
outdir = fullfile('outdir');

% image size is D1 x D2 pixels
D1 = 28;
D2 = 28;
D = D1 * D2;

% show N1 x N2 samples
N1 = 5;
N2 = 8;
N = N1 * N2;

%% -- mnist

load(fullfile(outdir, 'randomly_binarized_mnist.mat'), 'x_trn');
x = data_sample(x_trn, false, N);
samples = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        samples(ii,jj) = reshape(x(:,n), [D1,D2]);
    end
end

figure;
imshow(samples);
title('MNIST');

%% -- rbm

load(fullfile(outdir, 'rbm_CD25_500.mat'), 'rbm');

rbm.setGibbsState(double(rand(D, N) > 0.5));
x = rbm.gen(N, 2000);
samples = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        samples(ii,jj) = reshape(x(:,n), [D1,D2]);
    end
end

figure;
imshow(samples);
title('RBM');

%% -- nade

num_hidden = [1000, 750, 500, 250];

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    
    [~, x] = nade.gen(N);
    samples = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            samples(ii,jj) = reshape(x(:,n), [D1,D2]);
        end
    end

    figure;
    imshow(samples);
    title(sprintf('NADE, %d hiddens, kl divergence', i));
end

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', i)), 'nade');
    
    [~, x] = nade.gen(N);
    samples = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            samples(ii,jj) = reshape(x(:,n), [D1,D2]);
        end
    end

    figure;
    imshow(samples);
    title(sprintf('NADE, %d hiddens, square error', i));
end
