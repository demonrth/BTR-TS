function Out = tensor_contraction_ybz(X, Y, varargin)
% tensor_contraction_ybz Performs tensor contraction between two tensors X and Y
% along specified modes (dimensions).
%
% Syntax:
%   Out = tensor_contraction_ybz(X, Y, n, m)
%   Out = tensor_contraction_ybz(X, Y, Sx, Sy, n, m)
%
% Inputs:
%   X   - First tensor
%   Y   - Second tensor
%   n   - Mode (dimension) of tensor X to contract (scalar)
%   m   - Mode (dimension) of tensor Y to contract (scalar)
%   Sx  - Target order (number of dimensions) for tensor X (optional, defaults to ndims(X))
%   Sy  - Target order (number of dimensions) for tensor Y (optional, defaults to ndims(Y))
%
% Output:
%   Out - Resulting tensor after contracting X and Y along modes n and m
%         Dimensions of Out are all modes of X except n, combined with all modes of Y except m
%
% Description:
%   This function contracts (multiplies and sums over) the specified modes n of tensor X
%   and mode m of tensor Y. It reshapes and permutes the tensors to perform the contraction
%   as a matrix multiplication and then reshapes the result back into a tensor.
%
% Examples:
%   Out = tensor_contraction_ybz(X, Y, 3, 1);
%   % Contracts X along its 3rd mode with Y 
%
%   Out = tensor_contraction_ybz(X, Y, 3, 4, 3, 1);
%   % Contracts X (order 3) along its 3rd mode with Y (order 4) along its 1st mode 
%
%   Out = tensor_contraction_ybz(X, Y, [3,2], [1,2]);
%   % Contracts X's 3rd and 2nd modes with Y's 1st and 2nd modes,
%
%   Out = tensor_contraction_ybz(X, Y, 3, 4, [3,2], [1,2]);
%   % Contracts X (order 3) along its 3rd and 2nd modes with Y (order 4) along its 1st and 2nd modes.
%
%   [1] Yu-Bang Zheng, Ting-Zhu Huang*, Xi-Le Zhao*, Qibin Zhao, Tai-Xiang Jiang, 
%       "Fully-Connected Tensor Network Decomposition and Its Application to 
%       Higher-Order Tensor Completion", AAAI, 2021.
%   Created by Yu-Bang Zheng （zhengyubang@163.com）
%   Jun. 06, 2020


if nargin == 4
    % Case with 4 inputs: X, Y, n, m
    n = varargin{1};
    m = varargin{2};
    Sx = ndims(X);
    Sy = ndims(Y);
elseif nargin == 6
    % Case with 6 inputs: X, Y, Sx, Sy, n, m
    Sx = varargin{1};
    Sy = varargin{2};
    n = varargin{3};
    m = varargin{4};
else
    error('Incorrect number of input arguments. Use 4 or 6 inputs.');
end

Nx = ndims(X);
Ny = ndims(Y);
Lx = size(X);
Ly = size(Y);

% Pad dimensions with ones if actual order is less than target order
if Nx < Sx
    Lx = [Lx, ones(1, Sx - Nx)];
end
if Ny < Sy
    Ly = [Ly, ones(1, Sy - Ny)];
end

% Indices of modes except the contracting mode for X and Y
idxX = setdiff(1:Sx, n);
idxY = setdiff(1:Sy, m);

% Permute and reshape X: move contracting mode to last dimension
tmpX = permute(X, [idxX, n]);
tmpX = reshape(tmpX, prod(Lx(idxX)), prod(Lx(n)));

% Permute and reshape Y: move contracting mode to first dimension
tmpY = permute(Y, [m, idxY]);
tmpY = reshape(tmpY, prod(Ly(m)), prod(Ly(idxY)));

% Perform matrix multiplication (contraction)
tmpOut = tmpX * tmpY;

% Reshape output tensor back to combined non-contracted modes
Out = reshape(tmpOut, [Lx(idxX), Ly(idxY)]);
end
