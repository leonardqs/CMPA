clear

Is =  0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;
V = linspace(-1.95,0.7,200);
I1 = Is*(exp((1.2/0.025).*V)-1)+Gp.*V-Ib*(exp(-(1.2/0.025).*(V+Vb))-1);
I2 = I1 + 0.2*I1.*(rand(1,200)-0.5)*2;

x = V.';
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(x,I2',fo);
If = ff(x);


inputs = V.';   % y' transpose w cc, y.' transpose
targets = I2.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;



figure(1)
p4 = polyfit(V,I1,4);
fit4 = polyval(p4,V);
plot(V,I1);
hold on
plot(V,fit4);
hold on
p8 = polyfit(V,I1,8);
fit8 = polyval(p8,V);
plot(V,fit8);
hold on
plot(x,If)
hold on
plot(inputs, Inn)
hold on
%ylim([-0.25, 0.25])



