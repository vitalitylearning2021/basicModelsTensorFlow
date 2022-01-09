clear all
close all
clc

N = 100;

x = linspace(-1, 1, N);

y = 1 ./ (1+exp(-6.2 * x)); 

figure                 
set(gca, 'FontSize', 14)
plot(x, y, 'LineWidth', 2)
xlabel('x', 'FontSize', 14)
ylabel('y', 'FontSize', 14)