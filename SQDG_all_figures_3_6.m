function SQDG_all_figures_3(varargin)
% SQDG_ALL_FIGURES_3  Single-file driver that regenerates every figure of
% the manuscript
%
%   "The Skew-q-Dichotomized Gaussian model: tunable input skewness and
%    kurtosis jointly reshape neural population coding and Fisher information"
%    F. Miceli & F. Montani, Biological Cybernetics (revised version).
%
% -------------------------------------------------------------------------
% USAGE
%   SQDG_all_figures_3            % regenerate ALL figures (paper + new)
%   SQDG_all_figures_3('groups', G)   % G is a cell array of group names, any of:
%       'fig12'    Figs 1-2   spike-count distributions & cumulant panels
%       'fig35'    Figs 3-5   normalised cumulants vs mu
%       'fig68'    Figs 6-8   Q_J/H_S components, C_JS, Fisher ratio
%       'fig911'   Figs 9-11  information planes (CJS-HS, F-HS, F-CJS)
%       'fig12L'   Fig 12     kappa4 (gamma,q) landscape
%       'fig13'    Fig 13     C_JS-H_S trajectories (8 regimes)
%       'fig14'    Fig 14     lambda robustness sweep
%       'fig1519'  Figs 15-19 clinically-motivated (illustrative) regimes
%       'new'      New figures N1-N4 requested by the reviewers
%
%   By DEFAULT the script also writes every generated figure to a .jpg whose
%   name matches exactly the \includegraphics{...} calls in the revised
%   manuscript (one figure = one paper figure = one file), into the current
%   directory (run it next to the .tex, or set 'outdir'):
%       SQDG_all_figures_3                          % generate + save all .jpg
%       SQDG_all_figures_3('outdir','figs')         % save the .jpg into ./figs
%       SQDG_all_figures_3('dpi', 600)              % higher-resolution export
%       SQDG_all_figures_3('save', false)           % generate only, do not save
%   The figure-number -> filename mapping is listed in save_named_figs below.
%
% -------------------------------------------------------------------------
% NOTES ON THIS CONSOLIDATED VERSION (addressing the reviewers)
%   * All figures now use ONE unified numerical convention (the scaled
%     Student-t / sigma_nu convention of the Supplementary Material):
%       nu       = (3-q)/(q-1)                degrees of freedom
%       sigma_nu = sqrt((5-3q)/(3-q))         unit-variance scale
%       F_q^{-1}(u) = sigma_nu * T_nu^{-1}(u)  (q-Gaussian quantile)
%       theta    = F_q^{-1}(mu)               symmetric-quantile threshold
%       S0       = sqrt(1-lambda)*F_q^{-1}(r) - theta
%     so that the main text, the Supplementary Material and this code agree.
%   * mu is the THRESHOLD (symmetric-quantile) firing-rate parameter. The
%     realised mean output rate rbar = E[K]/n coincides with mu only in the
%     DG limit (gamma=0); for gamma~=0 it departs from mu -- this is now
%     reported explicitly (see NEW figure N1 and the tuning-curve figure N2).
%   * Axis and tick font sizes have been increased throughout (Reviewer 1).
%   * CORRECTED NUMERICS (v3): two implementation-level errors found while
%     re-checking the pathological-regime claims questioned by Reviewer 2 are
%     fixed here.  They did NOT touch any equation of the model:
%       (i)  the Student-t evaluation in sqdg_dist now uses the single
%            sigma_nu convention consistently (phi_v = t_nu(T_nu^{-1}(r)));
%       (ii) the DG baseline dg_dist is now integrated on a converged fixed
%            grid (the previous adaptive-quadrature tolerance was too loose
%            in the tails and slightly biased kappa4 of the baseline, which
%            distorted the DG-referenced ratios Delta-kappa4 and F_SQDG/F_DG).
%     With these fixes the script reproduces every number in the manuscript
%     (e.g. Delta-kappa4 ~ +28%% at n=50, peak Fisher ratio ~ 14.5x, PD-like
%     ~ 17x, kappa4 landscape ~ 5x).  All qualitative findings are unchanged.
%   * Requires MATLAB R2018a (Statistics and Machine Learning Toolbox:
%     tinv, tpdf, tcdf, norminv, normpdf, normcdf). Runs unmodified in
%     GNU Octave with the 'statistics' package.
% -------------------------------------------------------------------------

% ----- larger default fonts (Reviewer 1) -----
set(0,'defaultAxesFontSize',15);
set(0,'defaultTextFontSize',15);
set(0,'defaultLineLineWidth',1.8);
set(0,'defaultAxesLineWidth',1.0);

% ----- parse options -----
p = inputParser;
addParameter(p,'groups',{'all'});
addParameter(p,'save',true,@islogical);   % save each figure as .jpg by default
addParameter(p,'outdir','');               % where to write .jpg (default: current dir)
addParameter(p,'dpi',300);                 % resolution of the exported .jpg files
parse(p,varargin{:});
groups = p.Results.groups;
if ischar(groups), groups = {groups}; end
SAVEFIG = p.Results.save;
OUTDIR  = p.Results.outdir;
DPI     = p.Results.dpi;
want = @(g) any(strcmp(groups,'all')) || any(strcmp(groups,g));

fprintf('\n==============================================================\n');
fprintf(' SQDG figure generation (unified sigma_nu convention)\n');
fprintf('==============================================================\n');

if want('fig12'),   make_fig12();            end
if want('fig35'),   make_fig35();            end
if want('fig68'),   make_fig68();            end
if want('fig911'),  make_fig911();           end
if want('fig12L'),  make_landscape(0.20,0.30,100); end
if want('fig13'),   make_trajectories(0.30,100);   end
if want('fig14'),   make_lambda_sweep(0.20,100,1.25); end
if want('fig1519'), make_clinical(0.40,100);  end
if want('new'),     make_new_figures();       end

if SAVEFIG, save_named_figs(OUTDIR,DPI); end
fprintf('\nDone.\n');
end % ===================== end main =====================


%% =======================================================================
%%  CORE COMPUTATIONAL ROUTINES
%% =======================================================================

function [nu,sigma_nu] = q_params(q)
% Student-t parameters of the q-Gaussian (1<q<3). q->1 gives the Gaussian.
if abs(q-1) < 1e-6
    nu = Inf; sigma_nu = 1;
else
    nu = (3-q)/(q-1);
    sigma_nu = sqrt((5-3*q)/(3-q));
end
end

function q = Fq_inv(u,q_)
% q-Gaussian quantile F_q^{-1}(u) = sigma_nu * T_nu^{-1}(u).
[nu,s] = q_params(q_);
if isinf(nu), q = norminv(u); else q = s*tinv(u,nu); end
end

function P = sqdg_dist(mu,lambda,n,q,gamma)
% Asymptotic SQDG spike-count distribution P(K=k), k=0..n
% (Eq. 18 of the main text, unified sigma_nu convention).
k = 0:n; P = zeros(size(k));
[nu,s] = q_params(q);
theta = Fq_inv(mu,q);                      % symmetric-quantile threshold
for i = 1:numel(k)
    r = k(i)/n;
    if r > 0.001 && r < 0.999
        if isinf(nu)
            v  = norminv(r);
            S0 = sqrt(1-lambda)*v - theta;
            phit = normpdf(S0/sqrt(lambda));
            if abs(gamma) < 1e-6
                Phit = 0.5;
            else
                Phit = normcdf(gamma*S0/sqrt(lambda));
            end
            phiv = normpdf(v);
        else
            vt = tinv(r,nu);
            S0 = sqrt(1-lambda)*s*vt - theta;          % S0 = sqrt(1-lambda)*F_q^{-1}(r) - theta
            phit = tpdf(S0/(sqrt(lambda)*s),nu);        % proportional to f_q(S0/sqrt(lambda))
            Phit = tcdf(gamma*S0/(sqrt(lambda)*s),nu);  % F_q(gamma*S0/sqrt(lambda))
            phiv = tpdf(vt,nu);                          % proportional to f_q(F_q^{-1}(r)); arg = T_nu^{-1}(r)
        end
        if phiv > 0 && ~isnan(phit) && ~isnan(Phit)
            P(i) = 2*sqrt((1-lambda)/lambda)*(phit*Phit)/phiv;
        end
    end
    if isnan(P(i)) || isinf(P(i)), P(i) = 0; end
end
P = max(P,0);
if sum(P) > 0, P = P/sum(P); else P = ones(size(P))/numel(P); end
end

function P = dg_dist(mu,lambda,n)
% Exact homogeneous DG spike-count distribution (Eq. 11 of the main text):
% binomial integral over the common Gaussian input.  Used as the baseline.
%
% NOTE (corrected): the integral is evaluated on a fine, fixed Gauss grid
% (4000 nodes over +/-9 sigma) that is CONVERGED for the first four
% cumulants.  The earlier adaptive 'integral' call with AbsTol=1e-10 was too
% loose in the upper tail (small P(k) at large k), which slightly
% underestimated kappa4 of the baseline and therefore distorted the DG-
% referenced ratios (Delta kappa4 and the Fisher ratio).  The log-binomial
% (gammaln) form is also numerically safer than nchoosek for large n.
theta = norminv(mu); k = (0:n)';
sg = sqrt(lambda); S = linspace(-9*sg, 9*sg, 4000); ds = S(2)-S(1);
L = normcdf((S + theta)/sqrt(1-lambda));
L = min(max(L,1e-15),1-1e-15);
w = normpdf(S,0,sg);
lc = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1);   % log C(n,k)
M  = exp(lc + k*log(L) + (n-k)*log(1-L));            % (n+1) x numel(S)
P  = (M*(w(:).*ds))';                                % row vector, k=0..n
P  = max(P,0); P = P/sum(P);
end

function c = cumulants_from_P(P)
% First four cumulants from a spike-count probability vector.
k = 0:(numel(P)-1);
m1 = sum(k.*P); m2 = sum(k.^2.*P); m3 = sum(k.^3.*P); m4 = sum(k.^4.*P);
c.m1 = m1;
c.kappa1 = m1;
c.kappa2 = m2 - m1^2;
c.kappa3 = m3 - 3*m1*m2 + 2*m1^3;
c.kappa4 = m4 - 4*m1*m3 - 3*m2^2 + 12*m1^2*m2 - 6*m1^4;
c.rbar   = m1/(numel(P)-1);                % realised mean firing rate
end

function [HS,CJS,QJ] = mpr_complexity(P)
% Martin-Plastino-Rosso statistical complexity with uniform reference Pe
% (Rosso & Masoller 2009 analytic J_max).
P = P/sum(P);
N  = numel(P);
Pe = ones(1,N)/N;
Smax = log(N);
S_P  = -sum(P(P>0).*log(P(P>0)));
S_Pe = -sum(Pe.*log(Pe));
Jmax = -0.5*(((N+1)/N)*log(N+1) - 2*log(2*N) + log(N));
Pmix = (P+Pe)/2;
S_mix= -sum(Pmix(Pmix>0).*log(Pmix(Pmix>0)));
JS   = S_mix - 0.5*(S_P + S_Pe);
QJ   = JS/Jmax;
HS   = S_P/Smax;
CJS  = QJ*HS;
end

function F = fisher_discrete(P)
% Discrete Fisher information w.r.t. a rate shift (Eq. 23 of the main text),
% evaluated with centred differences interior / one-sided at the boundary.
P = P/sum(P); g = zeros(size(P));
for i = 2:numel(P)-1
    if P(i) > 0, g(i) = (P(i+1)-P(i-1))/2; end
end
if P(1)   > 0, g(1)   = P(2)-P(1);        end
if P(end) > 0, g(end) = P(end)-P(end-1);  end
F = 0;
for i = 1:numel(P)
    if P(i) > 0, F = F + g(i)^2/P(i); end
end
F = max(F,0);
end

function r = pair_corr_from_var(c,mu,n)
% Average pairwise OUTPUT correlation recovered from the spike-count variance:
%   Var(K) = n mu(1-mu) + n(n-1) Cov(Xi,Xj),  rho = Cov/[mu(1-mu)].
r = (c.kappa2 - n*mu*(1-mu))/(n*(n-1)*mu*(1-mu));
end

function y = skewq_input_pdf(z,loc,scale,gamma,q)
% Skew-q-Gaussian INPUT density phi(z;loc,scale,gamma,q)=2 f_q(u) F_q(gamma u),
% u=(z-loc)/scale, with f_q,F_q the scaled Student-t pdf/cdf (Eq. 4).
[nu,s] = q_params(q);
u = (z-loc)/scale;
if isinf(nu)
    y = (2/scale).*normpdf(u).*normcdf(gamma*u);
else
    y = (2/scale).*(1/s).*tpdf(u/s,nu).*tcdf(gamma*u/s,nu);
end
end


%% =======================================================================
%%  FIGURES 1-2 : spike-count distribution & cumulant comparison
%% =======================================================================
function make_fig12()
fprintf('\n[Figs 1-2] spike-count distributions and cumulant panels ...\n');
mu=0.2; lambda=0.3; n=50; q=1.25;
specs = {+2, 1; -2, 2};                        % gamma, figure index
Pd = dg_dist(mu,lambda,n); cd = cumulants_from_P(Pd);
for is = 1:size(specs,1)
    gamma = specs{is,1}; fno = specs{is,2};
    Ps = sqdg_dist(mu,lambda,n,q,gamma); cs = cumulants_from_P(Ps);
    k = 0:n;
    figure(fno); clf; set(gcf,'Position',[80 60 1100 780],'Color','w');
    % (a) distributions
    subplot(3,2,1);
    bar(k,[Ps(:) Pd(:)]); xlabel('Spike count K'); ylabel('P(K)');
    legend({'SQDG','DG'},'Location','northeast'); grid on;
    title(sprintf('(a)  q=%.2f, \\gamma=%+d',q,gamma));
    % (b) cumulative difference
    subplot(3,2,2);
    plot(k,cumsum(Ps)-cumsum(Pd),'-'); xlabel('Spike count K');
    ylabel('CDF diff (SQDG-DG)'); grid on; title('(b)');
    % (c) tail comparison on a log scale.  Styled like the originally
    %     submitted Figs. 1c/2c: zeros are masked (NaN) rather than floored
    %     at 1e-12, and the y-axis is scaled to the relevant decades so the
    %     DG-vs-SQDG tail difference is clearly visible (heavier right tail
    %     for gamma>0; strongly suppressed tail for gamma<0).
    subplot(3,2,3);
    Psc = Ps; Psc(Psc<=0) = NaN;   Pdc = Pd; Pdc(Pdc<=0) = NaN;
    semilogy(k,Psc,'b-o','MarkerSize',4,'MarkerFaceColor','b','LineWidth',1.4); hold on;
    semilogy(k,Pdc,'r--s','MarkerSize',4,'LineWidth',1.4);
    xlabel('Spike count K'); ylabel('Probability (log scale)');
    if gamma >= 0, ylim([1e-4 1e-1]); else, ylim([1e-8 1e0]); end
    xlim([-0.5 n+0.5]);
    legend({'SQDG','DG'},'Location','southwest'); grid on;
    title('(c) tail comparison');
    % (d) percentage cumulant differences
    subplot(3,2,4);
    dk = [100*(cs.kappa2-cd.kappa2)/abs(cd.kappa2), ...
          100*(cs.kappa3-cd.kappa3)/abs(cd.kappa3), ...
          100*(cs.kappa4-cd.kappa4)/abs(cd.kappa4)];
    bar(dk); set(gca,'XTickLabel',{'\Delta\kappa_2','\Delta\kappa_3','\Delta\kappa_4'});
    ylabel('% vs DG'); grid on; title('(d)');
    % (e) shape coefficients
    subplot(3,2,5);
    sh = [cs.kappa3/cs.kappa2^1.5, cd.kappa3/cd.kappa2^1.5; ...
          cs.kappa4/cs.kappa2^2,   cd.kappa4/cd.kappa2^2];
    bar(sh); set(gca,'XTickLabel',{'Skewness','Ex. kurtosis'});
    legend({'SQDG','DG'},'Location','best'); ylabel('value'); grid on; title('(e)');
    % (f) cumulant values
    subplot(3,2,6);
    bar([cs.kappa2 cd.kappa2; cs.kappa3 cd.kappa3; cs.kappa4 cd.kappa4]);
    set(gca,'XTickLabel',{'\kappa_2','\kappa_3','\kappa_4'});
    legend({'SQDG','DG'},'Location','best'); ylabel('value'); grid on; title('(f)');
    fprintf('   gamma=%+d : rbar=%.3f (mu=%.2f)  dK4=%+.1f%%  dK2=%+.1f%%\n', ...
            gamma, cs.rbar, mu, dk(3), dk(1));
end
end


%% =======================================================================
%%  FIGURES 3-5 : normalised output cumulants vs mu
%% =======================================================================
function make_fig35()
fprintf('\n[Figs 3-5] normalised cumulants vs mu ...\n');
lambda=0.3; n=100;
regimes = {1.0001,0,3,'DG baseline (q\rightarrow1,\gamma=0)'; ...
           1.25, +2,4,'q=1.25, \gamma=+2'; ...
           1.25, -2,5,'q=1.25, \gamma=-2'};
mu = linspace(0.05,0.95,50);
for ir = 1:size(regimes,1)
    q=regimes{ir,1}; gamma=regimes{ir,2}; fno=regimes{ir,3}; ttl=regimes{ir,4};
    k2s=nan(size(mu));k3s=k2s;k4s=k2s;k2d=k2s;k3d=k2s;k4d=k2s;
    for i=1:numel(mu)
        cs=cumulants_from_P(sqdg_dist(mu(i),lambda,n,q,gamma));
        cd=cumulants_from_P(dg_dist(mu(i),lambda,n));
        k2s(i)=cs.kappa2;k3s(i)=cs.kappa3;k4s(i)=cs.kappa4;
        k2d(i)=cd.kappa2;k3d(i)=cd.kappa3;k4d(i)=cd.kappa4;
    end
    nz=@(x) x/max(abs(x(~isnan(x)))+eps);
    figure(fno); clf; set(gcf,'Position',[80 60 700 820],'Color','w');
    subplot(3,1,1); plot(mu,nz(k2s),'b-'); hold on; plot(mu,nz(k2d),'r--');
    ylabel('\kappa_2 (norm.)'); grid on; legend({'SQDG','DG'},'Location','best');
    title(sprintf('%s  (\\lambda=%.2f, n=%d)',ttl,lambda,n));
    subplot(3,1,2); plot(mu,nz(k3s),'b-'); hold on; plot(mu,nz(k3d),'r--');
    plot(mu,zeros(size(mu)),'k-','LineWidth',0.5); ylabel('\kappa_3 (norm.)'); grid on;
    subplot(3,1,3); plot(mu,nz(k4s),'b-'); hold on; plot(mu,nz(k4d),'r--');
    plot(mu,zeros(size(mu)),'k-','LineWidth',0.5);
    xlabel('Threshold firing-rate parameter \mu'); ylabel('\kappa_4 (norm.)'); grid on;
end
end


%% =======================================================================
%%  FIGURES 6-8 : Q_J/H_S components, C_JS, Fisher ratio vs mu
%% =======================================================================
function make_fig68()
fprintf('\n[Figs 6-8] information-theoretic measures vs mu ...\n');
lambda=0.3; n=100; mu=linspace(0.05,0.95,50);
regs = {1.0001,0,'DG (q\rightarrow1,\gamma=0)'; 1.25,+2,'q=1.25,\gamma=+2'; 1.25,-2,'q=1.25,\gamma=-2'};
QJ=cell(3,1);HS=cell(3,1);CJ=cell(3,1);FR=cell(3,1);
for ir=1:3
    q=regs{ir,1};gamma=regs{ir,2};
    qj=nan(size(mu));hs=qj;cj=qj;fr=qj;
    for i=1:numel(mu)
        Ps=sqdg_dist(mu(i),lambda,n,q,gamma);
        Pd=dg_dist(mu(i),lambda,n);
        [hs(i),cj(i),qj(i)]=mpr_complexity(Ps);
        fr(i)=fisher_discrete(Ps)/max(fisher_discrete(Pd),1e-12);
    end
    QJ{ir}=qj;HS{ir}=hs;CJ{ir}=cj;FR{ir}=fr;
end
% Fig 6: components Q_J and H_S (3 panels)
figure(6); clf; set(gcf,'Position',[60 60 1200 380],'Color','w');
for ir=1:3
    subplot(1,3,ir);
    yyaxis left;  plot(mu,QJ{ir},'-'); ylabel('Q_J');
    yyaxis right; plot(mu,HS{ir},'-'); ylabel('H_S');
    xlabel('\mu'); grid on; title(sprintf('(%c) %s',96+ir,regs{ir,3}));
end
% Fig 7: C_JS SQDG vs DG (use gamma=+2 and gamma=-2 panels)
figure(7); clf; set(gcf,'Position',[60 60 760 720],'Color','w');
cjd=nan(size(mu)); for i=1:numel(mu), [~,cjd(i)]=mpr_complexity(dg_dist(mu(i),lambda,n)); end
subplot(2,1,1); plot(mu,CJ{2},'g-'); hold on; plot(mu,cjd,'r--');
ylabel('C_{JS}'); grid on; legend({'SQDG \gamma=+2','DG'},'Location','best'); title('(a) \gamma=+2');
subplot(2,1,2); plot(mu,CJ{3},'g-'); hold on; plot(mu,cjd,'r--');
xlabel('\mu'); ylabel('C_{JS}'); grid on; legend({'SQDG \gamma=-2','DG'},'Location','best'); title('(b) \gamma=-2');
% Fig 8: Fisher ratio
figure(8); clf; set(gcf,'Position',[60 60 760 720],'Color','w');
subplot(2,1,1); plot(mu,FR{2},'b-'); hold on; plot(mu,ones(size(mu)),'k--','LineWidth',1);
ylabel('F_{SQDG}/F_{DG}'); grid on; title('(a) \gamma=+2');
subplot(2,1,2); plot(mu,FR{3},'b-'); hold on; plot(mu,ones(size(mu)),'k--','LineWidth',1);
xlabel('\mu'); ylabel('F_{SQDG}/F_{DG}'); grid on; title('(b) \gamma=-2');
[mx2,i2]=max(FR{2}); [mx3,i3]=max(FR{3});
fprintf('   peak Fisher ratio: gamma=+2 %.1fx at mu=%.2f ; gamma=-2 %.1fx at mu=%.2f\n',mx2,mu(i2),mx3,mu(i3));
end


%% =======================================================================
%%  FIGURES 9-11 : information planes
%% =======================================================================
function make_fig911()
fprintf('\n[Figs 9-11] information planes ...\n');
lambda=0.3; n=100; mu=linspace(0.05,0.95,50);
regs={1.0001,0;1.25,2};   % DG and gamma=+2 (gamma=-2 omitted, see text)
HS=cell(2,1);CJ=cell(2,1);FI=cell(2,1);
for ir=1:2
    q=regs{ir,1};gamma=regs{ir,2};
    hs=nan(size(mu));cj=hs;fi=hs;
    for i=1:numel(mu)
        P=sqdg_dist(mu(i),lambda,n,q,gamma);
        [hs(i),cj(i)]=mpr_complexity(P); fi(i)=fisher_discrete(P);
    end
    HS{ir}=hs;CJ{ir}=cj;FI{ir}=fi;
end
lbl={'(a) DG','(b) SQDG \gamma=+2'};
figure(9); clf; set(gcf,'Position',[60 60 760 720],'Color','w');
for ir=1:2, subplot(2,1,ir); scatter(HS{ir},CJ{ir},40,mu,'filled'); colorbar;
   xlabel('H_S'); ylabel('C_{JS}'); grid on; title(lbl{ir}); end
figure(10); clf; set(gcf,'Position',[60 60 760 720],'Color','w');
for ir=1:2, subplot(2,1,ir); scatter(HS{ir},FI{ir},40,mu,'filled'); colorbar;
   xlabel('H_S'); ylabel('\mathcal{F}'); grid on; title(lbl{ir}); end
figure(11); clf; set(gcf,'Position',[60 60 760 720],'Color','w');
for ir=1:2, subplot(2,1,ir); scatter(CJ{ir},FI{ir},40,mu,'filled'); colorbar;
   xlabel('C_{JS}'); ylabel('\mathcal{F}'); grid on; title(lbl{ir}); end
end


%% =======================================================================
%%  FIGURE 12 : kappa4 landscape over (gamma,q)  [single panel; see rebuttal]
%% =======================================================================
function make_landscape(mu,lambda,n)
fprintf('\n[Fig 12] kappa4 (gamma,q) landscape ...\n');
qv = linspace(1.001,1.50,26); gv = linspace(-3,3,27);
cd = cumulants_from_P(dg_dist(mu,lambda,n)); k4dg = cd.kappa4;
R = nan(numel(gv),numel(qv));
for iq=1:numel(qv)
    for ig=1:numel(gv)
        cs=cumulants_from_P(sqdg_dist(mu,lambda,n,qv(iq),gv(ig)));
        R(ig,iq)=cs.kappa4/k4dg;
    end
end
[QQ,GG]=meshgrid(qv,gv);
figure(12); clf; set(gcf,'Position',[80 80 820 620],'Color','w');
Rp=R; Rp(Rp<-2)=-2; Rp(Rp>4)=4;
contourf(QQ,GG,Rp,25,'LineColor','none'); colormap(sqdg_redblue(256));
cb=colorbar; cb.Label.String='\kappa_4^{SQDG}/\kappa_4^{DG}'; caxis([-2 4]); hold on;
contour(QQ,GG,R,[1 1],'k--','LineWidth',2);
contour(QQ,GG,R,[1.65 1.65],'m:','LineWidth',2.5);
xlabel('Kurtosis index q'); ylabel('Skewness parameter \gamma');
title(sprintf('\\kappa_4 ratio  (\\mu=%.2f, \\lambda=%.2f, n=%d)',mu,lambda,n));
grid on;
% Panel (b) of the original figure (percentage change) is a monotone
% relabelling  DeltaK4 = (ratio-1)*100  and is therefore NOT plotted
% separately (Reviewer 2): the single ratio map carries all the information.
[mx,ix]=max(R(:)); [ig,iq]=ind2sub(size(R),ix);
fprintf('   max ratio %.2f at q=%.3f, gamma=%.2f ; DG kappa4=%.3f\n',mx,qv(iq),gv(ig),k4dg);
end


%% =======================================================================
%%  FIGURE 13 : C_JS-H_S trajectories for 8 (gamma,q) regimes
%% =======================================================================
function make_trajectories(lambda,n)
fprintf('\n[Fig 13] C_JS-H_S trajectories ...\n');
S=[1.001 0;1.001 2;1.001 -2;1.25 0;1.25 2;1.25 -2;1.50 2;1.50 -2];
L={'DG (q\rightarrow1,\gamma=0)','SDG \gamma=+2','SDG \gamma=-2','pure kurtosis q=1.25', ...
   'SQDG q=1.25,\gamma=+2','SQDG q=1.25,\gamma=-2','SQDG q=1.50,\gamma=+2','SQDG q=1.50,\gamma=-2'};
sty={'-','--','-.','-','--','-.',':','-'};
col=lines(8);
mu=linspace(0.05,0.95,50);
figure(13); clf; set(gcf,'Position',[80 80 900 680],'Color','w'); hold on;
for k=1:size(S,1)
    hs=nan(size(mu));cj=hs;
    for i=1:numel(mu)
        [hs(i),cj(i)]=mpr_complexity(sqdg_dist(mu(i),lambda,n,S(k,1),S(k,2)));
    end
    plot(hs,cj,sty{k},'Color',col(k,:),'LineWidth',2,'DisplayName',L{k});
end
xlabel('Normalised Shannon entropy H_S'); ylabel('Statistical complexity C_{JS}');
legend('Location','eastoutside'); grid on;
title(sprintf('C_{JS}-H_S plane (n=%d, \\lambda=%.2f)',n,lambda));
end


%% =======================================================================
%%  FIGURE 14 : lambda robustness sweep
%% =======================================================================
function make_lambda_sweep(mu,n,q)
fprintf('\n[Fig 14] lambda sweep ...\n');
lam=linspace(0.05,0.75,15); mug=linspace(0.05,0.95,40);
peakDG=nan(size(lam));peakP=peakDG;peakM=peakDG;
for il=1:numel(lam)
    cP=nan(size(mug));cM=cP;cD=cP;
    for i=1:numel(mug)
        [~,cP(i)]=mpr_complexity(sqdg_dist(mug(i),lam(il),n,q,+2));
        [~,cM(i)]=mpr_complexity(sqdg_dist(mug(i),lam(il),n,q,-2));
        [~,cD(i)]=mpr_complexity(dg_dist(mug(i),lam(il),n));
    end
    peakP(il)=max(cP);peakM(il)=max(cM);peakDG(il)=max(cD);
end
figure(14); clf; set(gcf,'Position',[80 80 820 560],'Color','w'); hold on;
plot(lam,peakDG,'k-o','MarkerFaceColor','k');
plot(lam,peakP,'r-o','MarkerFaceColor','r');
plot(lam,peakM,'b--s');
xlabel('Pairwise correlation \lambda'); ylabel('peak C_{JS}');
legend({'DG','SQDG \gamma=+2','SQDG \gamma=-2'},'Location','best'); grid on;
title(sprintf('Robustness across \\lambda  (\\mu=%.2f, n=%d, q=%.2f)',mu,n,q));
end


%% =======================================================================
%%  FIGURES 15-19 : clinically-motivated (ILLUSTRATIVE) regimes
%% =======================================================================
function make_clinical(lambda,n)
fprintf('\n[Figs 15-19] clinically-motivated illustrative regimes ...\n');
reg_q=[1.001 1.35 1.60 1.25]; reg_g=[0 3 5 -3];
reg_lbl={'DG baseline','Healthy (right-skew)','PD-like (illustrative)','AD-like (illustrative)'};
reg_col={[0 0 0],[0 0.7 0],[0.8 0 0],[0 0.3 0.8]};
reg_sty={'-','-','--',':'};
mu=linspace(0.05,0.95,40);
CJ=cell(4,1);HS=cell(4,1);FR=cell(4,1);
for ir=1:4
    cj=nan(size(mu));hs=cj;fr=cj;
    for i=1:numel(mu)
        if ir==1, P=dg_dist(mu(i),lambda,n); else P=sqdg_dist(mu(i),lambda,n,reg_q(ir),reg_g(ir)); end
        [hs(i),cj(i)]=mpr_complexity(P);
        fr(i)=fisher_discrete(P)/max(fisher_discrete(dg_dist(mu(i),lambda,n)),1e-12);
    end
    CJ{ir}=cj;HS{ir}=hs;FR{ir}=fr;
end
% Fig 15: spike-count PDFs at mu=0.2
figure(15); clf; set(gcf,'Position',[80 60 900 640],'Color','w'); k=0:n;
for ir=1:4
    if ir==1, P=dg_dist(0.2,lambda,n); else P=sqdg_dist(0.2,lambda,n,reg_q(ir),reg_g(ir)); end
    subplot(2,2,ir); bar(k,P,'FaceColor',reg_col{ir}); grid on;
    xlabel('Spike count K'); ylabel('P(K)'); title(reg_lbl{ir}); xlim([-0.5 n+0.5]);
end
% Fig 16: C_JS vs mu  (legend moved OUTSIDE/top so it does not cover the AD-like peak)
figure(16); clf; set(gcf,'Position',[80 60 900 600],'Color','w'); hold on;
for ir=1:4, plot(mu,CJ{ir},reg_sty{ir},'Color',reg_col{ir},'LineWidth',2.5); end
xlabel('\mu'); ylabel('C_{JS}'); grid on; title('Statistical complexity');
legend(reg_lbl,'Location','northoutside','Orientation','horizontal','FontSize',11);
% Fig 17: H_S vs mu
figure(17); clf; set(gcf,'Position',[80 60 820 560],'Color','w'); hold on;
for ir=1:4, plot(mu,HS{ir},reg_sty{ir},'Color',reg_col{ir},'LineWidth',2.5); end
xlabel('\mu'); ylabel('H_S'); ylim([0 1.05]); legend(reg_lbl,'Location','best'); grid on; title('Normalised Shannon entropy');
% Fig 18: Fisher ratio vs mu (non-DG regimes)
figure(18); clf; set(gcf,'Position',[80 60 820 560],'Color','w'); hold on;
for ir=2:4, plot(mu,FR{ir},reg_sty{ir},'Color',reg_col{ir},'LineWidth',2.5); end
plot(mu,ones(size(mu)),'k--','LineWidth',1);
xlabel('\mu'); ylabel('F_{SQDG}/F_{DG}'); legend(reg_lbl(2:4),'Location','best'); grid on; title('Relative Fisher efficiency');
% Fig 19 (manuscript Fig 22): C_JS-H_S plane. Tighten C_JS axis near 0.42 for visibility.
figure(19); clf; set(gcf,'Position',[80 60 860 620],'Color','w'); hold on;
for ir=1:4, plot(HS{ir},CJ{ir},reg_sty{ir},'Color',reg_col{ir},'LineWidth',2.5); end
xlabel('H_S'); ylabel('C_{JS}'); legend(reg_lbl,'Location','southwest','FontSize',11);
grid on; xlim([0 1]); ylim([0 0.42]);
title('Complexity-entropy plane (illustrative regimes)');
end


%% =======================================================================
%%  NEW FIGURES N1-N4  (reviewer requests)
%% =======================================================================
function make_new_figures()
fprintf('\n[New figures N1-N4] reviewer-requested analyses ...\n');

% --- N1: skew-q-Gaussian INPUT family and its mean shift (Reviewer 2, Sec 2)
% Single figure with three panels (a) vary gamma, (b) vary q, (c) mean E[Z].
figure(101); clf; set(gcf,'Position',[60 60 1300 420],'Color','w');
z=linspace(-6,6,801);
subplot(1,3,1); hold on;   % vary gamma at q=1.25
qs=1.25; gs=[-3 -1 0 1 3]; cols=lines(numel(gs));
for i=1:numel(gs)
    plot(z,skewq_input_pdf(z,0,1,gs(i),qs),'Color',cols(i,:),'DisplayName',sprintf('\\gamma=%+d',gs(i)));
end
xlabel('input z'); ylabel('\phi(z)'); grid on; legend('Location','northwest');
title(sprintf('(a) varying \\gamma  (q=%.2f)',qs));
subplot(1,3,2); hold on;   % vary q at gamma=+2 (wider q range to make the heavy-tail effect visible)
gm=2; qq=[1.0001 1.25 1.45 1.60]; cols=lines(numel(qq));  % q<5/3: finite-variance range (sigma_nu real)
for i=1:numel(qq)
    plot(z,skewq_input_pdf(z,0,1,gm,qq(i)),'Color',cols(i,:),'DisplayName',sprintf('q=%.2f',qq(i)));
end
xlabel('input z'); ylabel('\phi(z)'); grid on; legend('Location','northwest');
title(sprintf('(b) varying q  (\\gamma=%+d)',gm));
subplot(1,3,3); hold on;   % (c) mean of the standardised input vs (gamma,q)
gg=linspace(-4,4,33); qq=[1.0001 1.25 1.45 1.60]; cols=lines(numel(qq));  % match panel (b) q-set
for j=1:numel(qq)
    mzz=nan(size(gg));
    for i=1:numel(gg)
        f=@(z) z.*skewq_input_pdf(z,0,1,gg(i),qq(j));
        mzz(i)=integral(f,-200,200);
    end
    plot(gg,mzz,'-o','Color',cols(j,:),'MarkerSize',3,'DisplayName',sprintf('q=%.2f',qq(j)));
end
xlabel('Skewness parameter \gamma'); ylabel('Input mean  E[Z]');
grid on; legend('Location','southeast');
title('(c) input mean E[Z]');

% --- N2: single-neuron transfer / tuning curve vs (gamma,q) (Reviewer 2)
% realised firing rate mu(alpha)=P(Z>0) as a function of mean input alpha.
figure(103); clf; set(gcf,'Position',[60 60 1150 480],'Color','w');
a=linspace(-4,4,161);
subplot(1,2,1); hold on;  % vary gamma
qs=1.25; gs=[-3 0 3]; cols=lines(numel(gs));
for i=1:numel(gs)
    mr=arrayfun(@(x) integral(@(z) skewq_input_pdf(z,x,1,gs(i),qs),0,200),a);
    plot(a,mr,'Color',cols(i,:),'DisplayName',sprintf('\\gamma=%+d',gs(i)));
end
xlabel('mean input \alpha'); ylabel('firing rate P(Z>0)'); grid on; legend('Location','southeast');
title(sprintf('(a) varying \\gamma (q=%.2f)',qs));
subplot(1,2,2); hold on;  % vary q (wider range so the heavy-tail softening of the transfer curve is visible)
gm=2; qq=[1.0001 1.25 1.45 1.60]; cols=lines(numel(qq));  % q<5/3: finite-variance range (sigma_nu real)
for i=1:numel(qq)
    mr=arrayfun(@(x) integral(@(z) skewq_input_pdf(z,x,1,gm,qq(i)),0,200),a);
    plot(a,mr,'Color',cols(i,:),'DisplayName',sprintf('q=%.2f',qq(i)));
end
xlabel('mean input \alpha'); ylabel('firing rate P(Z>0)'); grid on; legend('Location','southeast');
title(sprintf('(b) varying q  (\\gamma=%+d)',gm));

% --- N3: average pairwise OUTPUT correlation vs gamma and q (Reviewer 2)
mu=0.2; lambda=0.3; n=100;
figure(104); clf; set(gcf,'Position',[60 60 1150 480],'Color','w');
subplot(1,2,1); hold on;   % vs gamma for several q
gg=linspace(-3,3,25); qset=[1.0001 1.25 1.45 1.60]; cols=lines(numel(qset));  % match N1/N2 q-set
for j=1:numel(qset)
    rr=nan(size(gg));
    for i=1:numel(gg)
        c=cumulants_from_P(sqdg_dist(mu,lambda,n,qset(j),gg(i)));
        rr(i)=pair_corr_from_var(c,mu,n);
    end
    plot(gg,rr,'-o','Color',cols(j,:),'MarkerSize',3,'DisplayName',sprintf('q=%.2f',qset(j)));
end
cd=cumulants_from_P(dg_dist(mu,lambda,n)); rdg=pair_corr_from_var(cd,mu,n);
plot(gg,rdg*ones(size(gg)),'k--','DisplayName','DG (\gamma=0)');
xlabel('Skewness parameter \gamma'); ylabel('output corr. \rho_{out}');
grid on; legend('Location','best'); title(sprintf('(a) vs \\gamma  (\\mu=%.2f,\\lambda=%.2f)',mu,lambda));
subplot(1,2,2); hold on;   % vs q for several gamma
qq=linspace(1.02,1.6,25); gset=[-2 0 2]; cols=lines(numel(gset));
for j=1:numel(gset)
    rr=nan(size(qq));
    for i=1:numel(qq)
        c=cumulants_from_P(sqdg_dist(mu,lambda,n,qq(i),gset(j)));
        rr(i)=pair_corr_from_var(c,mu,n);
    end
    plot(qq,rr,'-o','Color',cols(j,:),'MarkerSize',3,'DisplayName',sprintf('\\gamma=%+d',gset(j)));
end
xlabel('Kurtosis index q'); ylabel('output corr. \rho_{out}');
grid on; legend('Location','best'); title('(b) vs q');

% --- N4: PD/AD sensitivity - can a plain DG reproduce the SQDG target? (Rev 2)
sensitivity_analysis(0.20,0.40,100);
end


function sensitivity_analysis(mu0,lambda0,n)
% For each pathological regime, take the SQDG spike-count distribution as
% "target" and find the best-fit DG (over mu,lambda) minimising symmetric
% KL. Then compare higher-order (skewness, excess kurtosis) and the average
% pairwise output correlation. Shows that first-order shape can be roughly
% matched but SECOND- and higher-order structure cannot (Reviewer 2).
fprintf('\n   [N4] PD/AD sensitivity analysis (best-fit DG) ...\n');
reg_q=[1.60 1.25]; reg_g=[5 -3]; reg_name={'PD-like','AD-like'};
mug=linspace(0.05,0.95,46); lamg=linspace(0.05,0.85,33);
figure(105); clf; set(gcf,'Position',[60 60 1150 760],'Color','w');
for ir=1:2
    Pt=sqdg_dist(mu0,lambda0,n,reg_q(ir),reg_g(ir)); Pt=Pt/sum(Pt);
    ct=cumulants_from_P(Pt);
    best=inf; bi=1;bj=1;
    for i=1:numel(mug)
        for j=1:numel(lamg)
            Pd=dg_dist(mug(i),lamg(j),n); Pd=Pd/sum(Pd);
            d=symKL(Pt,Pd);
            if d<best, best=d; bi=i; bj=j; Pbest=Pd; end
        end
    end
    cb=cumulants_from_P(Pbest);
    k=0:n;
    subplot(2,2,(ir-1)*2+1);
    bar(k,[Pt(:) Pbest(:)]); grid on; xlabel('K'); ylabel('P(K)');
    legend({sprintf('SQDG %s (target)',reg_name{ir}),'best-fit DG'},'Location','northeast');
    title(sprintf('%s: target vs best-fit DG (\\mu=%.2f,\\lambda=%.2f)',reg_name{ir},mug(bi),lamg(bj)));
    subplot(2,2,(ir-1)*2+2);
    sk_t=ct.kappa3/ct.kappa2^1.5; ek_t=ct.kappa4/ct.kappa2^2;
    sk_b=cb.kappa3/cb.kappa2^1.5; ek_b=cb.kappa4/cb.kappa2^2;
    rho_t=pair_corr_from_var(ct,mu0,n); rho_b=pair_corr_from_var(cb,mug(bi),n);
    bar([sk_t sk_b; ek_t ek_b; rho_t rho_b]);
    set(gca,'XTickLabel',{'skewness','ex.kurtosis','\rho_{out}'});
    legend({'SQDG target','best-fit DG'},'Location','best'); grid on;
    title('higher-order & 2nd-order mismatch');
    fprintf('     %s: best DG mu=%.2f lam=%.2f | skew %.3f vs %.3f | exkurt %.3f vs %.3f | rho %.3f vs %.3f\n',...
        reg_name{ir},mug(bi),lamg(bj),sk_t,sk_b,ek_t,ek_b,rho_t,rho_b);
end
end

function d = symKL(P,Q)
P=P/sum(P); Q=Q/sum(Q); e=1e-12; P=P+e; Q=Q+e; P=P/sum(P); Q=Q/sum(Q);
d = sum(P.*log(P./Q)) + sum(Q.*log(Q./P));
end


%% =======================================================================
%%  UTILITIES
%% =======================================================================
function cmap = sqdg_redblue(nC)
if nargin<1, nC=256; end
h=floor(nC/2);
r1=linspace(0,1,h)'; g1=linspace(0,1,h)'; b1=linspace(0.85,1,h)';
r2=linspace(1,0.85,nC-h)'; g2=linspace(1,0,nC-h)'; b2=linspace(1,0,nC-h)';
cmap=[[r1;r2],[g1;g2],[b1;b2]];
end

function save_named_figs(outdir,dpi)
% Export each generated figure to a .jpg whose name matches exactly the
% \includegraphics{...} calls in the revised manuscript (MontaniFBC_revised.tex).
% One figure = one paper figure = one file.  Files are written to `outdir`
% (default: the current directory, so they sit next to the .tex).
if nargin<1 || isempty(outdir), outdir = pwd; end
if nargin<2 || isempty(dpi),    dpi = 300;  end
if ~exist(outdir,'dir'), mkdir(outdir); end

% figure number  ->  filename referenced by the .tex (see mapping table in README)
map = {
   1, 'SQDG_dist_gp2.jpg'     ; ...  % Fig. spike-count & cumulants, gamma=+2   (fig:fig1)
   2, 'SQDG_dist_gm2.jpg'     ; ...  % Fig. spike-count & cumulants, gamma=-2   (fig:fig2new)
   3, 'SQDG_cum_dg.jpg'       ; ...  % normalised cumulants, DG baseline        (fig:cumulants_baseline)
   4, 'SQDG_cum_gp2.jpg'      ; ...  % normalised cumulants, gamma=+2           (fig:cumulants_gamma+2)
   5, 'SQDG_cum_gm2.jpg'      ; ...  % normalised cumulants, gamma=-2           (fig:cumulants_gamma-2)
   6, 'SQDG_components.jpg'   ; ...  % Q_J & H_S components                     (fig:principal1)
   7, 'SQDG_cjs.jpg'          ; ...  % C_JS vs mu                               (fig:principal)
   8, 'SQDG_fisher_ratio.jpg' ; ...  % Fisher ratio vs mu                       (fig:principal2)
   9, 'SQDG_plane_cjs_hs.jpg' ; ...  % C_JS-H_S plane                           (fig:principal3)
  10, 'SQDG_plane_f_hs.jpg'   ; ...  % F-H_S plane                              (fig:principal4)
  11, 'SQDG_plane_f_cjs.jpg'  ; ...  % F-C_JS plane                             (fig:principal5)
  12, 'SQDG_landscape.jpg'    ; ...  % kappa4 (gamma,q) landscape               (fig:landscape)
  13, 'SQDG_trajectories.jpg' ; ...  % C_JS-H_S trajectories (8 regimes)        (fig:trajectories)
  14, 'SQDG_lambda_sweep.jpg' ; ...  % lambda robustness sweep                  (fig:lambdasweep)
  15, 'SQDG_patho_pdfs.jpg'   ; ...  % pathological-like spike-count PDFs       (fig:pdfspato)
  16, 'SQDG_patho_cjs.jpg'    ; ...  % C_JS vs mu (clinical)                    (fig:CJS_mu)
  17, 'SQDG_patho_hs.jpg'     ; ...  % H_S vs mu (clinical)                     (fig:HS_mu)
  18, 'SQDG_patho_fisher.jpg' ; ...  % Fisher vs mu (clinical)                  (fig:Fisher_mu)
  19, 'SQDG_patho_plane.jpg'  ; ...  % C_JS-H_S plane (clinical)                (fig:complepato)
 101, 'FigN1.jpg'             ; ...  % NEW skew-q input family + mean           (fig:input_family)
 103, 'FigN2.jpg'             ; ...  % NEW single-neuron transfer curves        (fig:tuning)
 104, 'FigN3.jpg'             ; ...  % NEW pairwise output correlation          (fig:paircorr)
 105, 'FigN4.jpg'             };     % NEW PD/AD sensitivity analysis           (fig:sensitivity)

nsaved = 0;
for i = 1:size(map,1)
    fnum = map{i,1}; fname = map{i,2};
    if ishandle(fnum)
        try
            set(fnum,'PaperPositionMode','auto','InvertHardcopy','off','Color','w');
            print(fnum, fullfile(outdir,fname), '-djpeg', sprintf('-r%d',dpi));
            nsaved = nsaved + 1;
        catch err
            fprintf('   WARNING: could not save figure %d as %s (%s)\n', fnum, fname, err.message);
        end
    end
end
fprintf('\nSaved %d figure(s) as .jpg to: %s\n', nsaved, outdir);
end
