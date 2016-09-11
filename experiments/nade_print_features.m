% Prints features of the rbm trained on binarized mnist, and of the various
% nades that were trained to mimic it.
%
% George Papamakarios, Jul 2015

clear;
close all;

% folder where to read results from
outdir = fullfile('outdir');

% image size is D1 x D2 pixels
D1 = 28;
D2 = 28;

% show N1 x N2 features
N1 = 5;
N2 = 8;

% greyscale range for plots
% if empty, adjusts according to image
range = [];

%% -- rbm

load(fullfile(outdir, 'rbm_CD25_500.mat'), 'rbm');

features = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        features(ii,jj) = reshape(rbm.W(:,n), [D1,D2]);
    end
end

fig = figure;
imshow(features, range);
title('RBM');

%% -- nade

num_hidden = [1000, 750, 500, 250];

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    
    features = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            features(ii,jj) = reshape([nade.W(:,n); 0], [D1,D2]);
        end
    end

    fig = figure;
    imshow(features, range);
    title(sprintf('NADE, %d hiddens, kl divergence', i));
end

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error.mat', i)), 'nade');
    
    features = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            features(ii,jj) = reshape([nade.W(:,n); 0], [D1,D2]);
        end
    end

    fig = figure;
    imshow(features, range);
    title(sprintf('NADE, %d hiddens, square error', i));
end
