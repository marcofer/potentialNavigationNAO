clear all;

freq = load("~/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/full_cycle_time.txt");
nofilt = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/nofiltp.txt");
filt = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/filtp.txt");
fps = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/cameraFPS.txt");
theta = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/theta.txt");
angularVel = load("/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot_hard_curve_vortex/angularVel.txt");

Fs = mean(freq)
Ts = 1.0/Fs;

t = [0:size(freq)-1]*Ts;

px = nofilt(:,1);
py = nofilt(:,2);
fpx = filt(:,1);
fpy = filt(:,2);

nofilt_norm = sqrt((px).*(px) + (py).*(py));
filt_norm = sqrt((fpx).*(fpx) + (fpy).*(fpy));

fontsize=50;

### Lowpass filter Bode Diagram ###
sys = tf([1],[1 1]);
[mag,pha,w] = bode(sys);
figure(7); 
hold on; 
grid;
title('Low-pass filter Gain Bode Diagram','fontsize',fontsize); 
ylabel('Gain [dB]','fontsize',25) ; 
xlabel('Frequency [rad/s]','fontsize',25); 
set(gca,'fontsize',fontsize);
semilogx(w,20*log10(mag),'linewidth',5);
print -deps -color lowpass.eps

### FFT ###
NFFT = length(nofilt);
Px = fft(px,NFFT);
Py = fft(py,NFFT);
Pnorm = fft(nofilt_norm,NFFT);

#Frequency vector
F = ((0:1/NFFT:1-1/NFFT)*Fs).';

#Magnitude vectors
magPx = abs(Px);
magPy = abs(Py);
magPnorm = abs(Pnorm);

figure(1);
h = figure(1);
plot(t,px,t,fpx,'r');hold on;
title("x-component of navigation vector p",'fontsize',fontsize);
#hl = legend("x not filtered", "x filtered");
#set(hl,'fontsize',fontsize);
axis([0 t(end) min(px) max(px)]);
set(gca,'fontsize',fontsize);
xlabel("Time [s]",'fontsize',fontsize);
ylabel("Magnitude [pixel]",'fontsize',fontsize);
grid;

print -deps -color softcurve_px.eps


figure(2);
plot(t,py,t,fpy,'r');hold on;
title("y-component of navigation vector p",'fontsize',fontsize);
axis([0 t(end) min(py) max(py)]);
set(gca,'fontsize',fontsize);
#hl2 = legend("y not filtered", "y filtered");
#set(hl2,'fontsize',fontsize);
xlabel("Time [s]",'fontsize',fontsize);
ylabel("Magnitude [pixel]",'fontsize',fontsize);
grid;

print -deps -color softcurve_py.eps


figure(3);
plot(t,nofilt_norm,t,filt_norm,'r');hold on;
title("Norm of navigation vector p",'fontsize',fontsize);
axis([0 t(end) min(nofilt_norm) max(nofilt_norm)]);
set(gca,'fontsize',fontsize);
#hl3 = legend("norm not filtered", "norm filtered");
#set(hl3,'fontsize',fontsize);
xlabel("Time [s]",'fontsize',fontsize);
ylabel("Magnitude [pixel]",'fontsize',fontsize);
grid;

print -deps -color softcurve_pnorm.eps

figure(4)
title('Magnitude diagram of the norm','fontsize',fontsize); hold on;
plot(F(1:NFFT/2),20*log10(magPnorm(1:NFFT/2)),'r');hold on;
xlabel('Frequency [Hz]','fontsize',fontsize)
ylabel('Magnitude [dB]','fontsize',fontsize)
set(gca,'fontsize',fontsize);
grid on;
axis tight 

print -deps -color softcurvemagPnorm.eps

figure(5)
title('Magnitude diagram of the x component','fontsize',fontsize); hold on;
plot(F(1:NFFT/2),20*log10(magPx(1:NFFT/2)),'r');hold on;
xlabel('Frequency [Hz]','fontsize',fontsize)
ylabel('Magnitude [dB]','fontsize',fontsize)
set(gca,'fontsize',fontsize);
grid on;
axis tight 

print -deps -color softcurvemagPx.eps

figure(6)
title('Magnitude diagram of the y component','fontsize',fontsize); hold on;
plot(F(1:NFFT/2),20*log10(magPy(1:NFFT/2)),'r');hold on;
xlabel('Frequency [Hz]','fontsize',fontsize)
ylabel('Magnitude [dB]','fontsize',fontsize)
set(gca,'fontsize',fontsize);
grid on;
axis tight 

print -deps -color softcurvemagPy.eps

#figure(3);
#plot(theta,'b');hold on;
#title("theta angle");
#legend("theta")
#grid;

#figure(4);
#plot(fps,'b');hold on;
#plot(freq,'r');hold on;
#title("working rates");
#legend("camera rate","loop rate (mean = %d Hz)",Tc);
#grid;

#figure(5);
#plot(angularVel,'b');hold on;
#title("angular velocity");
#legend("angular vel");
#grid;


pause();

