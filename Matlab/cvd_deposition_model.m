%% ============================================================================
%  박막 증착 공정 파라미터 모델링 (CVD/PVD)
%  Thin-Film Deposition Process Modeling - MATLAB version
% ============================================================================
%
%  목표:
%    1) 증착률(deposition rate)의 온도 의존성을 모델링한다.
%    2) Arrhenius 식으로부터 활성화에너지(Ea)를 추출한다.
%    3) 반응 율속(reaction-limited) / 물질전달 율속(mass-transport-limited)
%       영역을 구분하고 공정 윈도우(process window)를 도출한다.
%
%  이론 배경:
%    CVD 증착률은 온도에 따라 두 영역으로 나뉜다.
%
%    (1) 저온 - 반응 율속 영역 (reaction-limited / kinetic regime)
%          표면 화학반응이 율속 단계. 증착률은 Arrhenius 식을 따른다.
%              R = A * exp(-Ea / (k_B * T))
%          ln(R) vs 1/T 그래프에서 기울기 = -Ea/k_B  ->  Ea 추출 가능.
%
%    (2) 고온 - 물질전달 율속 영역 (mass-transport-limited / diffusion regime)
%          반응이 매우 빨라 반응물의 표면 공급(기상 확산)이 율속 단계.
%          온도 의존성이 약하다.  R ~ T^n (n = 1.5 ~ 2).
%
%  원본: Python 버전(cvd_deposition_model.py)을 MATLAB로 1:1 변환
%  주의: 변환 과정에서 일부 함수(curve_fit 등)는 실제 사용되지 않았던
%        부분이라 MATLAB 버전에서도 동일하게 생략했습니다. (확인됨)
% ============================================================================

clear; clc; close all;

%% ----------------------------------------------------------------------
%  물리 상수
% -------------------------------------------------------------------------
k_B_eV = 8.617333e-5;   % 볼츠만 상수 [eV/K]  (Ea를 eV 단위로 얻기 위함)

%% ============================================================================
%  STEP 1. 가상의 실험 데이터 생성
% ----------------------------------------------------------------------------
%  실제 프로젝트에서는 이 부분을 "측정한 실험값" 또는
%  논문에서 읽은 데이터로 교체하면 된다.
%  여기서는 학습을 위해, 알려진 Ea로 합성 데이터를 만들고
%  측정 노이즈를 약간 섞어 "실험처럼" 보이게 한다.
% ============================================================================

% 참값(ground truth) - 나중에 fit 결과와 비교하기 위함
Ea_true   = 0.85;        % 활성화에너지 [eV]  (전형적 CVD 값 0.5~2 eV 범위)
A_true    = 5.0e7;       % 전지수 인자(pre-exponential) [nm/min]
n_mtl     = 1.7;         % 물질전달 영역 온도 지수 R ~ T^n
R_mtl_ref = 120.0;       % 물질전달 영역 기준 증착률 스케일 [nm/min] (조정용)

% 온도 범위: 600 K ~ 1100 K (약 327 ~ 827 degC)
T = linspace(600, 1100, 24)';     % [K], 열벡터로 통일

% 반응 율속 영역의 증착률 (Arrhenius)
rate_reaction_limited = @(T, Ea, A) A .* exp(-Ea ./ (k_B_eV .* T));

% 물질전달 율속 영역의 증착률 (약한 온도 의존성)
rate_mass_transport = @(T, scale, n) scale .* (T ./ 1000.0) .^ n;

% 실제 증착률 = 두 메커니즘 중 '느린 쪽'이 율속(직렬 저항처럼 합성)
% 1/R_total = 1/R_rxn + 1/R_mtl  (두 단계가 직렬로 일어난다는 물리적 모델)
R_rxn = rate_reaction_limited(T, Ea_true, A_true);
R_mtl = rate_mass_transport(T, R_mtl_ref, n_mtl);
R_total_clean = 1.0 ./ (1.0 ./ R_rxn + 1.0 ./ R_mtl);

% 측정 노이즈 추가 (+-5% 가우시안) - 재현성 위해 시드 고정
rng(42);
noise = 1.0 + 0.05 .* randn(size(T));
R_measured = R_total_clean .* noise;

fprintf('%s\n', repmat('=', 1, 68));
fprintf('STEP 1. 실험 데이터 (가상)\n');
fprintf('%s\n', repmat('=', 1, 68));
fprintf('%8s %8s %8s %12s %8s\n', 'T [K]', 'T [C]', '1000/T', 'R [nm/min]', 'ln(R)');
for i = 1:length(T)
    fprintf('%8.0f %8.0f %8.3f %12.3f %8.3f\n', ...
        T(i), T(i)-273.15, 1000/T(i), R_measured(i), log(R_measured(i)));
end

%% ============================================================================
%  STEP 2. 반응 율속 영역만 골라 Arrhenius 피팅 -> Ea 추출
% ----------------------------------------------------------------------------
%  Arrhenius 분석은 '반응 율속' 영역(저온부)에서만 유효하다.
%  고온부(물질전달 영역)는 직선에서 벗어나므로 피팅에서 제외해야 한다.
%  여기서는 저온 절반(증착률이 낮은 영역)을 반응 율속 영역으로 본다.
% ============================================================================

% 1/T 와 ln(R)
invT = 1.0 ./ T;
lnR  = log(R_measured);

% 반응 율속 영역 선택: 저온부.
% 학습 포인트: 경계를 너무 높게(예: T<850K) 잡으면 물질전달 영향이
%   섞인 점까지 피팅에 포함되어 직선 기울기가 완만해지고 Ea가 과소평가된다.
%   순수 반응 율속 영역(저온부)만 엄격히 골라야 참 Ea에 가까워진다.
%   아래 T_boundary 값을 바꿔가며 Ea가 어떻게 변하는지 직접 확인해 볼 것.
T_boundary = 760;  % [K] 반응 율속/물질전달 경계 (조정 파라미터)
mask_rxn = T < T_boundary;
invT_fit = invT(mask_rxn);
lnR_fit  = lnR(mask_rxn);

% 선형 회귀: ln(R) = ln(A) - (Ea/k_B) * (1/T)
%   기울기 slope = -Ea/k_B  ->  Ea = -slope * k_B
p = polyfit(invT_fit, lnR_fit, 1);
slope = p(1);
intercept = p(2);
Ea_fit = -slope * k_B_eV;
A_fit  = exp(intercept);

% 결정계수 R^2 (피팅 품질)
lnR_pred = slope .* invT_fit + intercept;
ss_res = sum((lnR_fit - lnR_pred).^2);
ss_tot = sum((lnR_fit - mean(lnR_fit)).^2);
r_squared = 1 - ss_res / ss_tot;

fprintf('\n%s\n', repmat('=', 1, 68));
fprintf('STEP 2. Arrhenius 피팅 결과 (반응 율속 영역)\n');
fprintf('%s\n', repmat('=', 1, 68));
fprintf('  추출된 활성화에너지 Ea = %.4f eV   (참값: %.2f eV)\n', Ea_fit, Ea_true);
fprintf('  전지수 인자        A  = %.3e nm/min (참값: %.1e)\n', A_fit, A_true);
fprintf('  피팅 결정계수      R^2 = %.5f\n', r_squared);
fprintf('  피팅에 사용한 점 개수 = %d / %d\n', sum(mask_rxn), length(T));

%% ============================================================================
%  STEP 3. 시각화
% ============================================================================

fig = figure('Position', [100 100 1100 750]);
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- (a) 증착률 vs 온도 (선형 스케일) ---
ax1 = nexttile(tl, 1);
plot(ax1, T - 273.15, R_measured, 'o', 'Color', '#c0392b', ...
    'DisplayName', 'Measured (with noise)');
hold(ax1, 'on');
T_smooth = linspace(min(T), max(T), 300)';
R_smooth = 1.0 ./ (1.0 ./ rate_reaction_limited(T_smooth, Ea_true, A_true) ...
                  + 1.0 ./ rate_mass_transport(T_smooth, R_mtl_ref, n_mtl));
plot(ax1, T_smooth - 273.15, R_smooth, '-', 'Color', '#2c3e50', ...
    'LineWidth', 1.5, 'DisplayName', 'Model');
xlabel(ax1, 'Temperature [degC]');
ylabel(ax1, 'Deposition rate [nm/min]');
title(ax1, '(a) Deposition rate vs Temperature');
legend(ax1, 'Location', 'best');
grid(ax1, 'on');

% --- (b) Arrhenius plot: ln(R) vs 1000/T ---
ax2 = nexttile(tl, 2);
plot(ax2, 1000*invT(mask_rxn), lnR(mask_rxn), 'o', 'Color', '#c0392b', ...
    'DisplayName', 'Reaction-limited (used for fit)');
hold(ax2, 'on');
plot(ax2, 1000*invT(~mask_rxn), lnR(~mask_rxn), 's', ...
    'Color', '#7f8c8d', 'MarkerFaceColor', 'white', ...
    'DisplayName', 'Mass-transport (excluded)');
xfit = linspace(min(invT_fit), max(invT_fit), 50);
plot(ax2, 1000*xfit, slope*xfit + intercept, '--', 'Color', '#2980b9', ...
    'LineWidth', 1.8, 'DisplayName', sprintf('Fit: Ea = %.3f eV', Ea_fit));
xlabel(ax2, '1000 / T  [1/K]');
ylabel(ax2, 'ln( Deposition rate )');
title(ax2, '(b) Arrhenius plot -> slope gives Ea');
legend(ax2, 'Location', 'best');
grid(ax2, 'on');

% --- (c) 두 메커니즘의 기여 분리 ---
ax3 = nexttile(tl, 3);
semilogy(ax3, T - 273.15, rate_reaction_limited(T, Ea_true, A_true), '-', ...
    'Color', '#e67e22', 'DisplayName', 'Reaction-limited (Arrhenius)');
hold(ax3, 'on');
semilogy(ax3, T - 273.15, rate_mass_transport(T, R_mtl_ref, n_mtl), '-', ...
    'Color', '#27ae60', 'DisplayName', 'Mass-transport-limited');
semilogy(ax3, T - 273.15, R_total_clean, '-', 'Color', '#2c3e50', ...
    'LineWidth', 2, 'DisplayName', 'Combined (observed)');
xlabel(ax3, 'Temperature [degC]');
ylabel(ax3, 'Deposition rate [nm/min] (log)');
title(ax3, '(c) Rate-limiting mechanism separation');
legend(ax3, 'Location', 'best', 'FontSize', 8);
grid(ax3, 'on');

% --- (d) 공정 윈도우: 온도 민감도 (d lnR / dT) ---
% 온도에 둔감한(=재현성 좋은) 영역을 찾는다.
ax4 = nexttile(tl, 4);
dlnR_dT = gradient(log(R_total_clean), T);   % [1/K]
sensitivity_pct = dlnR_dT * 100;              % %/K 로 환산 (근사)
plot(ax4, T - 273.15, sensitivity_pct, '-', 'Color', '#8e44ad', 'LineWidth', 1.8);
hold(ax4, 'on');

stable = abs(sensitivity_pct) < 0.3;
if any(stable)
    yl = ylim(ax4);
    % stable 구간을 강조 (연속 구간이 아닐 수 있어 patch로 영역별 처리)
    stable_idx = find(stable);
    x_t = T - 273.15;
    patch(ax4, [x_t(stable_idx); flipud(x_t(stable_idx))], ...
          [repmat(yl(1), numel(stable_idx), 1); repmat(yl(2), numel(stable_idx), 1)], ...
          [0.18 0.8 0.44], 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
          'DisplayName', 'Stable window (<0.3 %/K)');

    T_window = T(stable);
    fprintf('\n%s\n', repmat('=', 1, 68));
    fprintf('STEP 3. 공정 윈도우 (Process Window)\n');
    fprintf('%s\n', repmat('=', 1, 68));
    fprintf('  온도 민감도 < 0.3 %%/K 인 안정 영역:\n');
    fprintf('    %.0f degC ~ %.0f degC (%.0f K ~ %.0f K)\n', ...
        min(T_window)-273.15, max(T_window)-273.15, min(T_window), max(T_window));
    fprintf('  -> 이 영역은 물질전달 율속이라 온도 변동에 둔감 = 재현성 우수\n');
end
yline(ax4, 0, 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
xlabel(ax4, 'Temperature [degC]');
ylabel(ax4, 'Rate sensitivity  [%/K]');
title(ax4, '(d) Process window (temperature sensitivity)');
legend(ax4, 'Location', 'best', 'FontSize', 8);
grid(ax4, 'on');

title(tl, 'Thin-Film Deposition Process Modeling (CVD)', ...
    'FontSize', 14, 'FontWeight', 'bold');

out_path = 'cvd_deposition_analysis_matlab.png';
exportgraphics(fig, out_path, 'Resolution', 150);
fprintf('\n[저장 완료] 그래프: %s\n', out_path);
