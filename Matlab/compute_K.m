clear
close all

% Find the spring as the 2-norm of the Hessian in the unstable equilibrium

% The functions are a sum of the contribution of the 4 different scatters
% in this cell. Since this is kinda complicated, they are defined below.
R_c = 3.3;
theta = pi/3; % lattice angle (60 = hexagonal)

% D_min
% D = R_c/sin(theta)
D = 4

L = [   D, cos(theta)*D
        0, sin(theta)*D ];

A = [   0, 1, 1, 0;
        0, 0, 1, 1 ];
S = L*A;

f = @(x) U_x(x, S, R_c);

% This is the unstable equilibrium at which the spring constant should be maximum
x0 = sum(L,2)/3
fx0 = f(x0)
dU2_separation = zeros(2,2);
for i=1:length(S(1,:))
    xS = x0 - S(:,i);
    if norm(xS) <= R_c
        dU2_separation = dU2_separation - d2_U_x(xS);
    end
end
dU2_separation

% Now, we have the sought for value...
K_max = norm(dU2_separation);

fprintf('K_max: %1.64f\n', K_max)


function [ U_x, dU_x, ddU_x ] = U_x(x, S, R_c)
    % This version can be used only on a single point
    % x 2x1
    % As it it too difficult to define it for multiple points.

    %%% Here, I define the functions used above. This is because in our case
    %%% the resulting values are a sum of the different contributions.
    U_r_inf = @(r) (1./(r.^6) - 2).*(1./(r.^6));
    U_R_c = U_r_inf(R_c);
    U_r = @(r) U_r_inf(r) - U_R_c;

    % Shift x to center it along every different scatter...
    dX = x - S;

    U_x = -sum(U_r(vecnorm(x))); % minus potential. d_U and dd_U already account for the -

    if(nargout > 1)
        d_U_x = @(x) full(x* spdiags( ( 12*(1 - 1./(vecnorm(x).^6)) .* (1./(vecnorm(x).^8)) )', 0, length(x(1,:)),length(x(1,:)) ));
        dU_x = sum( d_U_x(dX), 2 );
        if(nargout > 2)
            ddU_x = sum(d2_U_x(dX), 3);
        end
    end
end

function d2U = d2_U_x(x)
    [ m_x, n_x ] = size(x);
    x_3 = reshape(x,m_x,1,n_x);
    a = 12* (1 - 1./(vecnorm(x).^6)).*(1./(vecnorm(x).^8));
    a = reshape(a, 1,1,n_x);
    I_part = ones(m_x,m_x,n_x).*eye(m_x);
    I_part = pagemtimes(a,I_part);

    b = 12* (14*(1./(vecnorm(x).^6)) - 8 ).*(1./(vecnorm(x).^10));
    b = reshape(b, 1,1,n_x);

    X_part = pagemtimes( x_3, pagectranspose(x_3));
    X_part = pagemtimes(b,X_part);

    d2U = I_part+X_part;
end