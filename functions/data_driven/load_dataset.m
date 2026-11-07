function D = load_dataset(train_file, valid_file, d, a)
%LOAD_DATASET  Load a training pool and a validation set.
%
%   D = LOAD_DATASET(train_file, valid_file, d, a) reads the matrices
%   'Dataset' (training pool) and 'Validation' from the two .mat files.
%   Each column is one sample: rows 1:d hold the state x, row d+1 the
%   value V(x) and rows d+2:2d+1 the gradient grad V(x).
%
%   D has fields X, z, g (training) and Xv, zv, gv (validation), with one
%   sample per row, plus d and a.

S = load(train_file, 'Dataset');
T = load(valid_file, 'Validation');

D.d = d;
D.a = a(:).' .* ones(1, d);
D.X  = S.Dataset(1:d, :)';
D.z  = S.Dataset(d+1, :)';
D.g  = S.Dataset(d+2:2*d+1, :)';
D.Xv = T.Validation(1:d, :)';
D.zv = T.Validation(d+1, :)';
D.gv = T.Validation(d+2:2*d+1, :)';
end
