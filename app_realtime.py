import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ────────────────────── 모델 상수 (고정값) ──────────────────────
CONST = {
    "d_C": 100,       # CME 최적 곡률 반경 (nm)
    "d_V": 50,        # CvME 최적 곡률 반경 (nm)
    "kappa_C": 0.08,  # CME 막 굽힘 계수
    "kappa_V": 0.18,  # CvME 막 굽힘 계수
    "chi": 12.0,      # 곡률-전하 결합 상수
    "beta_pH": 0.5,   # pH 의존성 계수
    "pH": 7.4,        # 생리적 pH
    "T": 310.15,      # 생리적 온도 (K)
    "T_opt": 310.15,  # 최적 온도 (K)
    "k_p0": 0.05,     # 기준 내부화 속도 상수 (1/min)
    "sigma_kp": 5.0,  # 온도 감도 계수
    "q0": 0.0,        # 기준 전하 (추가)
}


# ────────────────────── 핵심 수식 ──────────────────────
def calc_alpha(d, q, C=CONST):
    """입자 직경 d와 전하 q에 대한 내부화 확률 α 계산"""
    q_eff = C["q0"] + C["beta_pH"] * (7.4 - C["pH"])
    dG_C = (C["kappa_C"] * (d - C["d_C"])**2 - C["chi"] * (q_eff + q)) * (C["T_opt"] / C["T"])
    dG_V = (C["kappa_V"] * (d - C["d_V"])**2 - C["chi"] * (q_eff + q)) * (C["T_opt"] / C["T"])

    def sigmoid(x):
        if x > 700: return 0.0
        if x < -700: return 1.0
        return 1.0 / (1.0 + np.exp(x))

    P_CME = sigmoid(dG_C)
    P_CvME = sigmoid(dG_V)
    alpha = P_CME + P_CvME - P_CME * P_CvME
    return {
        "P_CME": P_CME,
        "P_CvME": P_CvME,
        "alpha": alpha,
        "dG_C": dG_C,
        "dG_V": dG_V,
    }


def calc_kp(C=CONST):
    """온도 의존적 내부화 속도 상수 k_p(T)"""
    return C["k_p0"] * np.exp(-(C["T"] - C["T_opt"])**2 / (2 * C["sigma_kp"]**2))


def find_global_max(d_range, q_range, steps=150):
    """그리드 탐색으로 극댓값 (d*, q*, α_max) 찾기"""
    best = {"alpha": -1}
    d_min, d_max = d_range
    q_min, q_max = q_range
    for i in range(steps + 1):
        d = d_min + (d_max - d_min) * i / steps
        for j in range(steps + 1):
            q = q_min + (q_max - q_min) * j / steps
            a = calc_alpha(d, q)["alpha"]
            if a > best["alpha"]:
                best = {"alpha": a, "d": d, "q": q}
    return best


# ────────────────────── 시각화 함수 ──────────────────────
def build_alpha_grid(d_range, q_range, steps=150):
    """d, q 그리드 위에서 alpha 값을 계산해 행렬로 반환"""
    d_min, d_max = d_range
    q_min, q_max = q_range
    d_vals = np.linspace(d_min, d_max, steps)
    q_vals = np.linspace(q_min, q_max, steps)
    D, Q = np.meshgrid(d_vals, q_vals)
    alpha_grid = np.zeros_like(D)
    for i in range(steps):
        for j in range(steps):
            alpha_grid[i, j] = calc_alpha(D[i, j], Q[i, j])["alpha"]
    return d_vals, q_vals, alpha_grid


def plot_surface_3d(d_range, q_range, max_pt, current_pt=None, steps=150):
    """3D 공간 좌표(d, q, α)를 사용하는 plotly 인터랙티브 곡면"""
    d_vals, q_vals, alpha_grid = build_alpha_grid(d_range, q_range, steps)

    fig = go.Figure()

    # 메인 3D 곡면: x=d, y=q, z=alpha
    fig.add_trace(go.Surface(
        x=d_vals,
        y=q_vals,
        z=alpha_grid,
        colorscale='Plasma',
        opacity=0.95,
        showscale=True,
        colorbar=dict(title='내부화 확률'),
        contours=dict(
            z=dict(show=True, usecolormap=True,
                   highlightcolor="white", project_z=False)
        ),
        name="α(d, q)"
    ))

    # 극댓값 마커
    if max_pt and max_pt["alpha"] > 0:
        fig.add_trace(go.Scatter3d(
            x=[max_pt["d"]], y=[max_pt["q"]], z=[max_pt["alpha"]],
            mode='markers+text',
            marker=dict(size=6, color='yellow', symbol='diamond',
                        line=dict(color='white', width=1)),
            text=[f"d*={max_pt['d']:.1f}nm, q*={max_pt['q']:.2f}"],
            textposition='top center',
            textfont=dict(color='white', size=11),
            name="극댓값"
        ))

    # 현재 입력값 위치 마커
    if current_pt is not None:
        fig.add_trace(go.Scatter3d(
            x=[current_pt["d"]], y=[current_pt["q"]], z=[current_pt["alpha"]],
            mode='markers',
            marker=dict(size=6, color='#22c55e', symbol='circle',
                        line=dict(color='white', width=1)),
            name="현재 입력값"
        ))

    fig.update_layout(
        title=dict(text="내부화 확률", font=dict(size=16)),
        scene=dict(
            xaxis_title="직경 d (nm)",
            yaxis_title="전하 q (mV)",
            zaxis_title="내부화 확률",
            zaxis=dict(range=[0, 1]),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
    )
    return fig


def plot_time_curve(alpha, kp, t_max, current_t):
    """시간에 따른 내부화율 곡선"""
    t_vals = np.linspace(0, t_max, 200)
    uptake = alpha * (1 - np.exp(-kp * t_vals)) * 100
  
    fig = go.Figure()

    fig.add_trace(go.Scatter(
      x=t_vals,
      y=uptake,
      mode='lines',
      name='내부화율',
      line=dict(color='#22c55e', width=3)
    ))

    cur_val = alpha * (1 - np.exp(-kp * current_t)) * 100
    fig.add_trace(go.Scatter(
      x=[current_t],
      y=[cur_val],
      mode='markers',
      name=f't = {current_t} min',
      marker=dict(color='#facc15', size=12, line=dict(color='white', width=2))
    ))

    fig.add_vline(x=current_t, line_dash="dash", line_color="#facc15", opacity=0.5)
    fig.add_hline(y=cur_val, line_dash="dash", line_color="#facc15", opacity=0.3)

    fig.update_layout(
      xaxis_title="시간 t (min)",
      yaxis_title="내부화율 (%)",
      title="시간에 따른 누적 내부화율",
      font=dict(family="Malgun Gothic, sans-serif", size=12),
      hovermode='x',
      template='plotly_dark',
      height=400,
      margin=dict(l=40, r=20, t=50, b=40),
      xaxis=dict(range=[0, t_max], gridcolor='#334155'),
      yaxis=dict(range=[0, 105], gridcolor='#334155')
    )
    
    return fig


# ────────────────────── Streamlit 앱 (통합) ──────────────────────
st.set_page_config(page_title="나노입자 내부화 계산기", layout="wide")

# 사이드바
with st.sidebar:
    st.markdown("### 모델 상수 (고정값)")
    st.code("""
CME 최적 직경 = 100 nm
CvME 최적 직경 = 50 nm
CME 크기 미스매치 계수 = 0.08
CvME 크기 미스매치 계수 = 0.18
전하 영향력 계수 = 12.0
pH 전하 민감도 계수 = 0.5
절대 온도 = 310.15 K
최대 속도 상수 = 0.05 /min
""")
    st.caption("이 값들은 세포/환경의 물리적 특성을 나타냅니다.")

# 헤더
st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <h1 style="color:#000000;">나노입자 투과율 계산기</h1>
</div>
""", unsafe_allow_html=True)

# ─── 1. 입력 영역 (상단) ───
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h3 style="margin: 0; ">▼ 변수 입력</h3>
        <span style="font-size: 0.8rem; color: #94a3b8;">💡 원하는 값을 입력하면 모든 그래프와 수치가 자동으로 갱신됩니다.</span>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    d = st.number_input("직경 (d)", min_value=20.0, max_value=200.0, value=100.0, step=1.0)
with col2:
    q = st.number_input("표면 전하 (q)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
with col3:
    t = st.slider("경과 시간 (t)", min_value=0, max_value=210, value=30, step=1)

# ─── 2. 즉시 계산 ───
res = calc_alpha(d, q)
kp = calc_kp()
cur_uptake = res["alpha"] * (1 - np.exp(-kp * t)) * 100

# 상단 요약 카드
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:12px; padding:15px 20px; margin:10px 0 20px; border-left:5px solid #818cf8;">
    <div style="display:flex; justify-content:space-around; flex-wrap:wrap; text-align:center;">
        <div><span style="color:#94a3b8;">최대 내부화 확률 (α)</span><br><b style="color:#818cf8; font-size:28px;">{res['alpha']:.6f}</b></div>
        <div><span style="color:#94a3b8;">속도 상수 (kₚ)</span><br><b style="color:#a5b4fc; font-size:28px;">{kp:.6f} /min</b></div>
        <div><span style="color:#94a3b8;">현재 내부화율 (t = {t}min)</span><br><b style="color:#facc15; font-size:28px;">{cur_uptake:.3f} %</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 3. 상세 수치 (접기) ───
with st.expander("에너지 장벽 및 경로 확률 상세 보기", expanded=False):
    c1, c2 = st.columns(2)
    c1.metric("ΔG_C (CME)", f"{res['dG_C']:.4f}")
    c2.metric("ΔG_V (CvME)", f"{res['dG_V']:.4f}")
    c1.metric("P_CME", f"{res['P_CME']:.6f}")
    c2.metric("P_CvME", f"{res['P_CvME']:.6f}")
    st.info(f"α = (CME 내부화 확률) + (CvME 내부화 확률) − (CME 내부화 확률)·(CvME 내부화 확률) = **{res['alpha']:.6f}**")

# ─── 4. 그래프 영역 ───
st.markdown("---")
st.markdown("### ▼ 실시간 그래프")

# 곡면 탐색 범위 자동 설정 (입력값 중심)
d_range = [max(20, d - 50), min(200, d + 50)]
q_range = [max(-20, q - 3), min(20, q + 3)]
max_pt = find_global_max(d_range, q_range, 150)
current_pt = {"d": d, "q": q, "alpha": res["alpha"]}

st.subheader("① 시간 함수")
fig_time = plot_time_curve(res["alpha"], kp, 210, t)
st.plotly_chart(fig_time, use_container_width=True)

st.subheader("② 3D 확률 공간 좌표계")
st.caption(
    f"마우스로 드래그하면 회전, 스크롤로 확대 / 축소됩니다.",
    unsafe_allow_html=True
)
fig_surf = plot_surface_3d(d_range, q_range, max_pt, current_pt)
st.plotly_chart(fig_surf, use_container_width=True)
