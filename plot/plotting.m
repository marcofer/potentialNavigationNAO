clear all;

freq = load("~/Documents/Software/qi_work_tree/potentialNavigationNAO/plot18/full_cycle_time.txt");
nofilt = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot18/nofiltp.txt");
filt = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot18/filtp.txt");
fps = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot18/cameraFPS.txt");
theta = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot18/theta.txt");

figure(1);
plot(nofilt(:,1),'b');hold on;
plot(filt(:,1),'r');hold on;
legend("x not filtered", "x filtered");

figure(2);
plot(nofilt(:,2),'b');hold on;
plot(filt(:,2),'r');hold on;
legend("y not filtered", "y filtered");

figure(3);
plot(theta,'b');hold on;
legend("theta")

figure(4);
plot(fps,'b');hold on;
plot(freq,'r');hold on;
legend("camera rate","loop rate");
text (10, 30, "loop rate mean",mean(freq), "mean(freq)");

mean(freq)

pause();

