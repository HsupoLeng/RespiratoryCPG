%Example:
clc
clear all
T = 150;
I_current = 10;
N = 10;

% The initial condition y_ini is a 4*N-by-1 column vector
y_ini = [rand(N,1) ; rand(N,1);  rand(N,1);  0.2*rand(N,1)];


%%
close all
clc
dt = 0.1;
Isyn = @(t) 0;

[t, y] = ode45(@(t, y) nHHModel(t, y, I_current, Isyn), [0 T], y_ini);

V = y(:,    1:N )';
m = y(:,  N+1:2*N)';
n = y(:,2*N+1:3*N)';
h = y(:,3*N+1:4*N)';
plot(t, V)
hold off


%% This section is unfinished
clc

dt = 0.1;
V = [y_ini(      1:N,:)];
m = [y_ini(  N+1:2*N,:)]; 
n = [y_ini(2*N+1:3*N,:)];
h = [y_ini(3*N+1:4*N,:)];
t = [];

for ti = 0:dt:T 
    
    [to, y] = ode45(@(t, y) nHHModel(t, y, I_current, Isyn), [ti  ti+dt], y_ini);
    
    t = [t; to(2:end)];
    
    % Note that the output y is a length(t)-by-4*N matrix, the following
    % code convet it to N-by-length(t) for each variable.
    V = [V, y(2:end,    1:N )' ];
    m = [m, y(2:end,  N+1:2*N)'];
    n = [n, y(2:end,2*N+1:3*N)'];
    h = [h, y(2:end,3*N+1:4*N)'];
    %plot(t, V)
    
    % Other things need to be implemented
    %1) loc findpeak(V) and then keep trajing of t_peak
    %2) Select the t_peak that is NOT to far away from current time t
    %3) Creat a function looks like I_peak = @(t) sum( exp((-t-t_peak)./tau) )
    
    
    % Set the initial condition y_ini for the next loop,
    % it's a 4N-by-1 column vector
    y_ini = [V(:,end) ; m(:,end); n(:,end); h(:,end)];
    
end

%%
figure(1)
plot(t, V)
hold off