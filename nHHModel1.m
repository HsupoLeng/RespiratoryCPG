function [ dydt ] = nHHModel(t, y, I_in, Isyn)
% HHModel(t, y, I_in) compute the ifferential equations the Hodgkin-Huxley model.
%   Function returns the time differential of variables V, m, n and h.
%
%   Example:
%     T = 150;
%     I_current = 10;
%     N = 10;  % Number of neurons
%
%     y_ini = [rand(N,1) ; rand(N,1);  rand(N,1);  0.2*rand(N,1)];
%     [t, y] = ode45(@(t, y) nHHModel(t, y, I_current), [0  T], y_ini);
% 
%     V = y(:,    1:N );
%     m = y(:,  N+1:2*N);
%     n = y(:,2*N+1:3*N);
%     h = y(:,3*N+1:4*N);
%
%     plot(t,V)


% Constrent Parameters
C  = 1;    % (1 muF/cm^2)

% Get the variables for previous time step
N = size(y,1)/4;   % Number of cells
V = y(    1:N,  :);
m = y(  N+1:2*N,:);
n = y(2*N+1:3*N,:);
h = y(3*N+1:4*N,:);


% Evaluate the external inputs at time t.
Isyn_t = feval(Isyn, t);

[I_ions, dmdt, dndt, dhdt] = I_L_Na_L(V, m, n, h);

dVdt = (- I_ions + I_in )./C;

% (- I_ions + I_in)./C;

dydt = [dVdt; dmdt; dndt; dhdt];

end


function [ I_ions, dmdt, dndt, dhdt ] = I_L_Na_L(V, m, n, h)

% Constrent Parameters
gL = 0.3;  % ( mS/cm^2)
gNa= 120;
gK = 36;

VL = 10.613;
VNa= 115;
VK = -12;

% Ion currents (N-by-1 column vectors for N neurons)
I_ions = gL*(V - VL) + gNa*m.^3.*h.*(V - VNa) + gK.*n.^4.*(V - VK);

% Ion gating variables 
am = (-0.1.*V + 2.5)./(exp(-0.1.*V +2.5)-1);    bm = 4.*exp(-V./18);
an = (-0.01.*V + 0.1)./(exp(-0.1.*V + 1)-1);    bn = 0.125.*exp(-V./80);
ah = (0.07).*exp(-V./20);                       bh = 1./(exp(-V./10 + 3)+1);

dmdt = am.*(1-m) - bm.*m;
dndt = an.*(1-n) - bn.*n;
dhdt = ah.*(1-h) - bh.*h;


end
