"""
============================================================================
박막 증착 공정 파라미터 모델링 (CVD/PVD)
Thin-Film Deposition Process Modeling
============================================================================

목표:
  1) 증착률(deposition rate)의 온도 의존성을 모델링한다.
  2) Arrhenius 식으로부터 활성화에너지(Ea)를 추출한다.
  3) 반응 율속(reaction-limited) / 물질전달 율속(mass-transport-limited)
     영역을 구분하고 공정 윈도우(process window)를 도출한다.

이론 배경:
  CVD 증착률은 온도에 따라 두 영역으로 나뉜다.

  (1) 저온 - 반응 율속 영역 (reaction-limited / kinetic regime)
        표면 화학반응이 율속 단계. 증착률은 Arrhenius 식을 따른다.
            R = A * exp(-Ea / (k_B * T))
        ln(R) vs 1/T 그래프에서 기울기 = -Ea/k_B  →  Ea 추출 가능.

  (2) 고온 - 물질전달 율속 영역 (mass-transport-limited / diffusion regime)
        반응이 매우 빨라 반응물의 표면 공급(기상 확산)이 율속 단계.
        온도 의존성이 약하다.  R ∝ T^n (n ≈ 1.5 ~ 2).

  실제 공정에서는 재현성이 좋은 물질전달 영역을 선호하는 경우가 많다
  (작은 온도 변동에 증착률이 둔감하기 때문).

작성: 박막 공정 학습 프로젝트 (입문)
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# matplotlib 한글 폰트가 없을 수 있으므로 영문 라벨 사용
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------------
# 물리 상수
# ----------------------------------------------------------------------------
k_B_eV = 8.617333e-5   # 볼츠만 상수 [eV/K]  (Ea를 eV 단위로 얻기 위함)

# ============================================================================
# STEP 1. 가상의 실험 데이터 생성
# ----------------------------------------------------------------------------
# 실제 프로젝트에서는 이 부분을 "측정한 실험값" 또는
# 논문에서 읽은 데이터로 교체하면 된다.
# 여기서는 학습을 위해, 알려진 Ea로 합성 데이터를 만들고
# 측정 노이즈를 약간 섞어 "실험처럼" 보이게 한다.
# ============================================================================

# 참값(ground truth) — 나중에 fit 결과와 비교하기 위함
Ea_true   = 0.85        # 활성화에너지 [eV]  (전형적 CVD 값 0.5~2 eV 범위)
A_true    = 5.0e7       # 전지수 인자(pre-exponential) [nm/min]
n_mtl     = 1.7         # 물질전달 영역 온도 지수 R ∝ T^n
R_mtl_ref = 120.0       # 물질전달 영역 기준 증착률 스케일 [nm/min] (조정용)

# 온도 범위: 600 K ~ 1100 K (약 327 ~ 827 °C)
T = np.linspace(600, 1100, 24)          # [K]

def rate_reaction_limited(T, Ea, A):
    """반응 율속 영역의 증착률 (Arrhenius)"""
    return A * np.exp(-Ea / (k_B_eV * T))

def rate_mass_transport(T, scale, n):
    """물질전달 율속 영역의 증착률 (약한 온도 의존성)"""
    return scale * (T / 1000.0) ** n

# 실제 증착률 = 두 메커니즘 중 '느린 쪽'이 율속(직렬 저항처럼 합성)
# 1/R_total = 1/R_rxn + 1/R_mtl  (두 단계가 직렬로 일어난다는 물리적 모델)
R_rxn = rate_reaction_limited(T, Ea_true, A_true)
R_mtl = rate_mass_transport(T, R_mtl_ref, n_mtl)
R_total_clean = 1.0 / (1.0 / R_rxn + 1.0 / R_mtl)

# 측정 노이즈 추가 (±5% 가우시안) — 재현성 위해 시드 고정
rng = np.random.default_rng(42)
noise = rng.normal(1.0, 0.05, size=T.shape)
R_measured = R_total_clean * noise

print("=" * 68)
print("STEP 1. 실험 데이터 (가상)")
print("=" * 68)
print(f"{'T [K]':>8} {'T [C]':>8} {'1000/T':>8} {'R [nm/min]':>12} {'ln(R)':>8}")
for Ti, Ri in zip(T, R_measured):
    print(f"{Ti:8.0f} {Ti-273.15:8.0f} {1000/Ti:8.3f} {Ri:12.3f} {np.log(Ri):8.3f}")

# ============================================================================
# STEP 2. 반응 율속 영역만 골라 Arrhenius 피팅 → Ea 추출
# ----------------------------------------------------------------------------
# Arrhenius 분석은 '반응 율속' 영역(저온부)에서만 유효하다.
# 고온부(물질전달 영역)는 직선에서 벗어나므로 피팅에서 제외해야 한다.
# 여기서는 저온 절반(증착률이 낮은 영역)을 반응 율속 영역으로 본다.
# ============================================================================

# 1/T 와 ln(R)
invT = 1.0 / T
lnR = np.log(R_measured)

# 반응 율속 영역 선택: 저온부.
# ※ 학습 포인트: 경계를 너무 높게(예: T<850K) 잡으면 물질전달 영향이
#   섞인 점까지 피팅에 포함되어 직선 기울기가 완만해지고 Ea가 과소평가된다.
#   순수 반응 율속 영역(저온부)만 엄격히 골라야 참 Ea에 가까워진다.
#   아래 T_boundary 값을 바꿔가며 Ea가 어떻게 변하는지 직접 확인해 볼 것.
T_boundary = 760  # [K] 반응 율속/물질전달 경계 (조정 파라미터)
mask_rxn = T < T_boundary
invT_fit = invT[mask_rxn]
lnR_fit = lnR[mask_rxn]

# 선형 회귀: ln(R) = ln(A) - (Ea/k_B) * (1/T)
#   기울기 slope = -Ea/k_B  →  Ea = -slope * k_B
slope, intercept = np.polyfit(invT_fit, lnR_fit, 1)
Ea_fit = -slope * k_B_eV
A_fit = np.exp(intercept)

# 결정계수 R^2 (피팅 품질)
lnR_pred = slope * invT_fit + intercept
ss_res = np.sum((lnR_fit - lnR_pred) ** 2)
ss_tot = np.sum((lnR_fit - np.mean(lnR_fit)) ** 2)
r_squared = 1 - ss_res / ss_tot

print("\n" + "=" * 68)
print("STEP 2. Arrhenius 피팅 결과 (반응 율속 영역)")
print("=" * 68)
print(f"  추출된 활성화에너지 Ea = {Ea_fit:.4f} eV   (참값: {Ea_true} eV)")
print(f"  전지수 인자        A  = {A_fit:.3e} nm/min (참값: {A_true:.1e})")
print(f"  피팅 결정계수      R² = {r_squared:.5f}")
print(f"  피팅에 사용한 점 개수 = {mask_rxn.sum()} / {len(T)}")

# ============================================================================
# STEP 3. 시각화
# ============================================================================

fig = plt.figure(figsize=(13, 9))

# --- (a) 증착률 vs 온도 (선형 스케일) ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(T - 273.15, R_measured, "o", color="#c0392b", label="Measured (with noise)")
T_smooth = np.linspace(T.min(), T.max(), 300)
R_smooth = 1.0 / (1.0 / rate_reaction_limited(T_smooth, Ea_true, A_true)
                  + 1.0 / rate_mass_transport(T_smooth, R_mtl_ref, n_mtl))
ax1.plot(T_smooth - 273.15, R_smooth, "-", color="#2c3e50", lw=1.5, label="Model")
ax1.set_xlabel("Temperature [°C]")
ax1.set_ylabel("Deposition rate [nm/min]")
ax1.set_title("(a) Deposition rate vs Temperature")
ax1.legend()

# --- (b) Arrhenius plot: ln(R) vs 1000/T ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(1000 * invT[mask_rxn], lnR[mask_rxn], "o", color="#c0392b",
         label="Reaction-limited (used for fit)")
ax2.plot(1000 * invT[~mask_rxn], lnR[~mask_rxn], "s", color="#7f8c8d",
         markerfacecolor="white", label="Mass-transport (excluded)")
# 피팅 직선
xfit = np.linspace(invT_fit.min(), invT_fit.max(), 50)
ax2.plot(1000 * xfit, slope * xfit + intercept, "--", color="#2980b9", lw=1.8,
         label=f"Fit: Ea = {Ea_fit:.3f} eV")
ax2.set_xlabel("1000 / T  [1/K]")
ax2.set_ylabel("ln( Deposition rate )")
ax2.set_title("(b) Arrhenius plot → slope gives Ea")
ax2.legend()

# --- (c) 두 메커니즘의 기여 분리 ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.semilogy(T - 273.15, rate_reaction_limited(T, Ea_true, A_true), "-",
             color="#e67e22", label="Reaction-limited (Arrhenius)")
ax3.semilogy(T - 273.15, rate_mass_transport(T, R_mtl_ref, n_mtl), "-",
             color="#27ae60", label="Mass-transport-limited")
ax3.semilogy(T - 273.15, R_total_clean, "-", color="#2c3e50", lw=2,
             label="Combined (observed)")
ax3.set_xlabel("Temperature [°C]")
ax3.set_ylabel("Deposition rate [nm/min] (log)")
ax3.set_title("(c) Rate-limiting mechanism separation")
ax3.legend(fontsize=8)

# --- (d) 공정 윈도우: 온도 민감도 (d lnR / dT) ---
# 온도에 둔감한(=재현성 좋은) 영역을 찾는다.
ax4 = fig.add_subplot(2, 2, 4)
dlnR_dT = np.gradient(np.log(R_total_clean), T)   # [1/K]
sensitivity_pct = dlnR_dT * 100                    # %/K 로 환산 (근사)
ax4.plot(T - 273.15, sensitivity_pct, "-", color="#8e44ad", lw=1.8)
# 민감도 < 0.3%/K 인 영역을 '안정 공정 윈도우'로 표시
stable = np.abs(sensitivity_pct) < 0.3
if stable.any():
    ax4.fill_between(T - 273.15, sensitivity_pct.min(), sensitivity_pct.max(),
                     where=stable, color="#2ecc71", alpha=0.2,
                     label="Stable window (<0.3 %/K)")
    T_window = T[stable]
    print("\n" + "=" * 68)
    print("STEP 3. 공정 윈도우 (Process Window)")
    print("=" * 68)
    print(f"  온도 민감도 < 0.3 %/K 인 안정 영역:")
    print(f"    {T_window.min()-273.15:.0f} °C ~ {T_window.max()-273.15:.0f} °C "
          f"({T_window.min():.0f} K ~ {T_window.max():.0f} K)")
    print(f"  → 이 영역은 물질전달 율속이라 온도 변동에 둔감 = 재현성 우수")
ax4.axhline(0, color="k", lw=0.5)
ax4.set_xlabel("Temperature [°C]")
ax4.set_ylabel("Rate sensitivity  [%/K]")
ax4.set_title("(d) Process window (temperature sensitivity)")
ax4.legend(fontsize=8)

fig.suptitle("Thin-Film Deposition Process Modeling (CVD)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = "/home/claude/cvd_deposition_analysis.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"\n[저장 완료] 그래프: {out_path}")
