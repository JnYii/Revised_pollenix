from dataclasses import dataclass
import math
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 京粉报 BJ Pollenix — 北京动物园 / 颐和园 双公园版
#
# 模型定位：
# 这是一个 literature-grounded experimental risk index，
# 不是临床诊断工具，也不是官方医学风险分级。
#
# 主要文献：
# 1) Zhou Y, Dai J, Liu H, Liu X. (2022)
#    Tourist risk assessment of pollen allergy in tourism attractions:
#    A case study in the Summer Palace, Beijing, China.
#    Frontiers in Public Health 10:1030066.
#    DOI: 10.3389/fpubh.2022.1030066
#
# 2) 北京动物园乔木数量来自：
#    《城市公园树木安全风险评估——以北京动物园为例》
#    DOI: 10.12171/j.1000-1522.20210200
#
# 本版本相较旧版：
# - 删除“公园面积 -> 蒿属/草本密度”的经验公式。
# - 删除证据较弱的 UV 与连续晴天模块。
# - 以 I_GZA / Green Zone Allergenicity Index 为植被危险度基础。
# - 使用文献中的木本花粉浓度 5 级阈值。
# - 风、温、湿、降雨、日照保留为动态气象修正层。
# - PM2.5 仅作为实验性炎症增强层。
# - 侧边栏可一键切换“北京动物园 / 颐和园”。
#
# 用户设定的有效绿地：
# - 北京动物园：40 ha
# - 颐和园：总面积 290.8 ha 的 40% = 116.32 ha
# ============================================================


@dataclass(frozen=True)
class AllergenicTree:
    chinese_name: str
    scientific_name: str
    count: int
    allergenic_potential: float   # AP
    pollen_emission: float        # PE
    flowering_duration: float     # FD, day
    crown_area: float             # S, m²
    canopy_height: float          # H, m
    parameter_note: str = ""


@dataclass(frozen=True)
class ParkProfile:
    name: str
    total_area_ha: float
    effective_green_area_ha: float
    trees: tuple
    data_note: str
    reliability_note: str


@dataclass
class ParkEnvironment:
    date: str

    pollen_density: float
    # 粒/m³
    # 北京市区域背景花粉浓度；本模型主要使用 0–60 的输入范围

    wind_speed: float
    # m/s

    humidity: float
    # %

    rainfall_48h: float
    # mm

    sunlight_hours: float
    # h

    temperature: float
    # °C

    pm25: float
    # μg/m³


# ============================================================
# 颐和园论文 Table 4 参数
# ============================================================

SUMMER_PALACE_TOTAL_AREA_HA = 290.8
SUMMER_PALACE_EFFECTIVE_GREEN_AREA_HA = (
    SUMMER_PALACE_TOTAL_AREA_HA * 0.40
)

SUMMER_PALACE_TREES = (
    AllergenicTree(
        "侧柏",
        "Platycladus orientalis",
        9446,
        3,
        3,
        21,
        13.72,
        9.6,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "圆柏",
        "Sabina chinensis / Juniperus chinensis",
        4062,
        3,
        3,
        16,
        15.20,
        10.1,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "垂柳",
        "Salix babylonica",
        3536,
        2,
        2,
        16,
        44.40,
        16.1,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "油松",
        "Pinus tabuliformis",
        3205,
        1,
        2,
        11,
        95.03,
        13.4,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "毛白杨",
        "Populus tomentosa",
        682,
        2,
        2,
        20,
        94.99,
        17.8,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "桑树",
        "Morus alba",
        196,
        2,
        1,
        20,
        67.02,
        14.2,
        "Zhou et al. (2022), Table 4",
    ),
    AllergenicTree(
        "白蜡",
        "Fraxinus chinensis",
        114,
        4,
        3,
        16,
        94.99,
        11.8,
        "Zhou et al. (2022), Table 4",
    ),
)


# ============================================================
# 北京动物园数据
#
# 乔木普查数量：
# 侧柏 325
# 圆柏 889
# 绦柳 317
# 油松 599
# 毛白杨 314
#
# 北京动物园没有公开与颐和园 Table 4 完全同维度的
# AP / PE / FD / crown area / canopy height 数据。
#
# 因此这里：
# - 同种植物直接使用颐和园 Table 4 参数作为 species-level proxy；
# - 绦柳 Salix matsudana 'Pendula' 暂以 Salix babylonica 参数代理。
#
# 这意味着动物园 I_GZA 是“代理估计”，不是原论文实测 I_GZA。
# ============================================================

BEIJING_ZOO_TOTAL_AREA_HA = 86.0
BEIJING_ZOO_EFFECTIVE_GREEN_AREA_HA = 40.0

BEIJING_ZOO_TREES = (
    AllergenicTree(
        "侧柏",
        "Platycladus orientalis",
        325,
        3,
        3,
        21,
        13.72,
        9.6,
        "数量来自北京动物园乔木普查；其余参数借用颐和园同种数据",
    ),
    AllergenicTree(
        "圆柏",
        "Juniperus chinensis",
        889,
        3,
        3,
        16,
        15.20,
        10.1,
        "数量来自北京动物园乔木普查；其余参数借用颐和园同种/同义种数据",
    ),
    AllergenicTree(
        "绦柳",
        "Salix matsudana 'Pendula'",
        317,
        2,
        2,
        16,
        44.40,
        16.1,
        "数量来自北京动物园乔木普查；AP/PE/FD/S/H 暂以 Salix babylonica 代理",
    ),
    AllergenicTree(
        "油松",
        "Pinus tabuliformis",
        599,
        1,
        2,
        11,
        95.03,
        13.4,
        "数量来自北京动物园乔木普查；其余参数借用颐和园同种数据",
    ),
    AllergenicTree(
        "毛白杨",
        "Populus tomentosa",
        314,
        2,
        2,
        20,
        94.99,
        17.8,
        "数量来自北京动物园乔木普查；其余参数借用颐和园同种数据",
    ),
)


PARKS = {
    "北京动物园": ParkProfile(
        name="北京动物园",
        total_area_ha=BEIJING_ZOO_TOTAL_AREA_HA,
        effective_green_area_ha=BEIJING_ZOO_EFFECTIVE_GREEN_AREA_HA,
        trees=BEIJING_ZOO_TREES,
        data_note=(
            "园区总面积按 86 ha；有效绿地按用户设定的 40 ha。"
            "乔木数量来自北京动物园乔木普查。"
        ),
        reliability_note=(
            "动物园目前只有树种数量是园区实测值；"
            "AP、PE、花期、冠幅和树高暂借用颐和园相同/近缘树种参数，"
            "因此 I_GZA 为代理估计。"
        ),
    ),
    "颐和园": ParkProfile(
        name="颐和园",
        total_area_ha=SUMMER_PALACE_TOTAL_AREA_HA,
        effective_green_area_ha=SUMMER_PALACE_EFFECTIVE_GREEN_AREA_HA,
        trees=SUMMER_PALACE_TREES,
        data_note=(
            "园区总面积 290.8 ha；有效绿地按整园 40% 计算，"
            f"即 {SUMMER_PALACE_EFFECTIVE_GREEN_AREA_HA:.2f} ha。"
        ),
        reliability_note=(
            "主要树种数量、AP、PE、花期、冠幅和树高均来自 "
            "Zhou et al. (2022) Table 4。"
        ),
    ),
}


# ============================================================
# 基础工具函数
# ============================================================

def clamp(value, minimum=0.0, maximum=100.0):
    return max(
        minimum,
        min(value, maximum)
    )


def gaussian_unit(
    value,
    optimal,
    sigma
):
    return math.exp(
        -((value - optimal) ** 2)
        / (2 * sigma ** 2)
    )


def logistic_unit(
    value,
    midpoint,
    steepness
):
    return 1.0 / (
        1.0
        + math.exp(
            -steepness
            * (value - midpoint)
        )
    )


def geometric_mean(values):
    positive_values = [
        max(float(v), 1e-6)
        for v in values
    ]

    log_mean = (
        sum(
            math.log(v)
            for v in positive_values
        )
        / len(positive_values)
    )

    return math.exp(log_mean)


# ============================================================
# I_GZA
#
# Zhou et al. (2022):
#
# I_GZA =
# [ Σ(N_i × AP_i × PE_i × FD_i × S_i × H_i) ]
# / (378 × S_T)
#
# N  = individual number
# AP = allergenic potential
# PE = pollen emission
# FD = flowering duration
# S  = crown projection area
# H  = canopy height
# ST = assessment-unit area (m²)
#
# canonical_igza:
#   严格按总园区面积作为 ST。
#
# effective_green_igza:
#   BJ Pollenix 扩展指标，把用户定义的“有效绿地面积”
#   作为局部植被源面积进行归一化。
#   它不是 Zhou et al. 原文的标准 I_GZA，故记作 I_GZA*。
# ============================================================

def igza_component(
    tree: AllergenicTree
):
    return (
        tree.count
        * tree.allergenic_potential
        * tree.pollen_emission
        * tree.flowering_duration
        * tree.crown_area
        * tree.canopy_height
    )


def calculate_igza(
    park: ParkProfile,
    area_ha: float
):
    numerator = sum(
        igza_component(tree)
        for tree in park.trees
    )

    area_m2 = (
        area_ha
        * 10000.0
    )

    return (
        numerator
        / (
            378.0
            * area_m2
        )
    )


def species_hazard_shares(
    park: ParkProfile
):
    raw = {
        tree.chinese_name:
        igza_component(tree)
        for tree in park.trees
    }

    total = max(
        sum(raw.values()),
        1e-9
    )

    return {
        name:
        100.0 * value / total
        for name, value in raw.items()
    }


# 颐和园有效绿地 I_GZA* 作为内部归一化参考。
SUMMER_PALACE_EFFECTIVE_IGZA = (
    calculate_igza(
        PARKS["颐和园"],
        PARKS["颐和园"].effective_green_area_ha
    )
)


def vegetation_score(
    effective_igza
):
    """
    以颐和园 I_GZA* 作为 100 分内部参考。
    这只是 BJ Pollenix 的跨公园相对比较尺度，
    不是医学概率或 Zhou et al. 的风险分界。
    """
    ratio = (
        effective_igza
        / max(
            SUMMER_PALACE_EFFECTIVE_IGZA,
            1e-9
        )
    )

    return clamp(
        ratio * 100.0
    )


# ============================================================
# 背景花粉浓度评分
#
# 恢复原 BJ Pollenix 的连续型背景花粉算法：
#
# score = 100 / (1 + exp[-0.158 × (pollen_density - 24)])
#
# 输入范围主要设为 0–60 粒/m³。
# 这样背景花粉浓度会连续影响总风险，而不是被压成固定等级。
# ============================================================

def pollen_score(
    pollen_density
):
    return (
        100.0
        * logistic_unit(
            pollen_density,
            midpoint=24.0,
            steepness=0.158
        )
    )


# ============================================================
# 春季木本花粉季节因子
#
# 颐和园论文实证重点是春季 3-5 月。
# 为了让日期连续变化，用 4 月为中心做平滑 Gaussian。
#
# peak=4.0, sigma=0.85 是 BJ Pollenix 平滑参数，
# 不是 Zhou et al. 原论文回归系数。
# ============================================================

def circular_month_distance(
    month,
    peak_month
):
    diff = abs(
        month - peak_month
    )

    return min(
        diff,
        12 - diff
    )


def spring_season_factor(
    month
):
    distance = (
        circular_month_distance(
            month,
            4.0
        )
    )

    factor = (
        0.03
        + 0.97
        * math.exp(
            -(distance ** 2)
            / (
                2
                * 0.85 ** 2
            )
        )
    )

    return min(
        max(factor, 0.0),
        1.0
    )


# ============================================================
# 气象动态层
#
# 以下方向由北京花粉-气象研究支持：
# - 中等风速利于局地悬浮/传播，过大风速后局地浓度下降；
# - 春季约 5-20°C 更利于高花粉天气；
# - 降雨具有清除作用；
# - 湿度关系非线性；
# - 日照与花粉物候/释放相关。
#
# 具体 sigma / midpoint 是 BJ Pollenix 的平滑实现参数，
# 并非声称为临床阈值。
# ============================================================

def wind_condition_score(
    wind_speed
):
    return (
        100.0
        * gaussian_unit(
            wind_speed,
            optimal=3.0,
            sigma=1.6
        )
    )


def temperature_condition_score(
    temperature
):
    return (
        100.0
        * gaussian_unit(
            temperature,
            optimal=17.0,
            sigma=7.0
        )
    )


def humidity_condition_score(
    humidity
):
    return (
        100.0
        * gaussian_unit(
            humidity,
            optimal=55.0,
            sigma=20.0
        )
    )


def rainfall_condition_score(
    rainfall_48h
):
    # 降雨越多，空气中花粉越容易被清除。
    return (
        100.0
        * math.exp(
            -rainfall_48h
            / 12.0
        )
    )


def sunlight_condition_score(
    sunlight_hours
):
    return (
        100.0
        * logistic_unit(
            sunlight_hours,
            midpoint=6.0,
            steepness=0.35
        )
    )


def pm25_score(
    pm25
):
    return (
        100.0
        * logistic_unit(
            pm25,
            midpoint=75.0,
            steepness=0.03
        )
    )


# ============================================================
# 风险计算
#
# 结构：
# 1) 背景空气花粉等级 -> pollen score
# 2) I_GZA* -> local vegetation score
# 3) 日期 -> spring seasonal activation
# 4) 气象条件 -> weather score
# 5) PM2.5 -> experimental inflammatory modifier
#
# 为减少任意人工权重，天气内部采用几何平均，
# 暴露层采用背景花粉与季节性植被危险度的几何耦合。
#
# PM2.5 最大只提供约 10% 的实验性增强。
# 这仍是 BJ Pollenix 的模型设计，不是已发表临床公式。
# ============================================================

def calculate_risk(
    env: ParkEnvironment,
    park: ParkProfile
):
    date_obj = datetime.strptime(
        env.date,
        "%Y-%m-%d"
    )

    current_month = (
        date_obj.month
        + (date_obj.day - 1)
        / 30.0
    )

    canonical_igza = (
        calculate_igza(
            park,
            park.total_area_ha
        )
    )

    effective_igza = (
        calculate_igza(
            park,
            park.effective_green_area_ha
        )
    )

    V = vegetation_score(
        effective_igza
    )

    season = (
        spring_season_factor(
            current_month
        )
    )

    seasonal_V = (
        V
        * season
    )

    P = pollen_score(
        env.pollen_density
    )


    W = wind_condition_score(
        env.wind_speed
    )

    T = temperature_condition_score(
        env.temperature
    )

    H = humidity_condition_score(
        env.humidity
    )

    R = rainfall_condition_score(
        env.rainfall_48h
    )

    S = sunlight_condition_score(
        env.sunlight_hours
    )

    M = pm25_score(
        env.pm25
    )

    weather_score = geometric_mean([
        W,
        T,
        H,
        R,
        S
    ])

    # 背景花粉是直接观测/输入的空气暴露；
    # 植被危险度是局部源强修正。
    exposure_score = geometric_mean([
        max(P, 1.0),
        max(seasonal_V, 1.0)
    ])

    # 天气只调制释放/传播条件，不直接制造花粉。
    base_risk = (
        exposure_score
        * (
            0.60
            + 0.40
            * weather_score
            / 100.0
        )
    )

    # 实验性 PM2.5 炎症增强，最大约 +10%。
    pm_multiplier = (
        1.0
        + 0.10
        * M
        / 100.0
    )

    total_risk = clamp(
        base_risk
        * pm_multiplier
    )

    if total_risk < 20:
        level = "低风险"
        recommendation = "一般可正常活动"
    elif total_risk < 40:
        level = "轻度风险"
        recommendation = "敏感人群可适当缩短户外停留"
    elif total_risk < 60:
        level = "中度风险"
        recommendation = "敏感人群建议减少长时间户外活动"
    elif total_risk < 80:
        level = "高风险"
        recommendation = "过敏人群建议佩戴口罩并减少停留"
    else:
        level = "极高风险"
        recommendation = "过敏人群建议尽量避免长时间进入园区"

    species_shares = (
        species_hazard_shares(
            park
        )
    )

    total_known_trees = sum(
        tree.count
        for tree in park.trees
    )

    tree_density = (
        total_known_trees
        / park.effective_green_area_ha
    )

    factors = {
        "背景花粉": P,
        "局部植被": seasonal_V,
        "风速条件": W,
        "温度条件": T,
        "湿度条件": H,
        "降雨条件": R,
        "日照条件": S,
        "PM2.5": M,
    }

    return {
        "风险评分": round(
            total_risk,
            2
        ),
        "风险等级": level,
        "活动建议": recommendation,
        "背景花粉评分": round(
            P,
            2
        ),
        "标准 I_GZA": round(
            canonical_igza,
            4
        ),
        "有效绿地 I_GZA*": round(
            effective_igza,
            4
        ),
        "植被相对评分": round(
            V,
            2
        ),
        "春季活跃系数": round(
            season,
            3
        ),
        "季节性植被评分": round(
            seasonal_V,
            2
        ),
        "天气适宜度": round(
            weather_score,
            2
        ),
        "PM2.5增强系数": round(
            pm_multiplier,
            3
        ),
        "已纳入树木数量": total_known_trees,
        "有效绿地树木密度": round(
            tree_density,
            2
        ),
        "物种危险贡献": {
            k: round(v, 2)
            for k, v
            in species_shares.items()
        },
        "详细指标": {
            k: round(v, 2)
            for k, v
            in factors.items()
        },
    }


# ============================================================
# Streamlit UI — Minimal Apple-inspired redesign
# ============================================================

st.set_page_config(
    page_title="京粉报 · BJ Pollenix",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 只保留轻量 CSS：
# - 系统字体
# - 浅灰背景
# - 白色圆角 metric 卡片
# - 轻微阴影
# 不再用大量嵌套 HTML card，避免元素重叠/穿模。
st.markdown(
    """
<style>
html, body, [class*="css"]{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Text",
        "SF Pro Display",
        "PingFang SC",
        "Helvetica Neue",
        Arial,
        sans-serif;
}

.stApp{
    background:#F5F5F7;
}

.block-container{
    max-width:1180px;
    padding-top:2rem;
    padding-bottom:3rem;
}

section[data-testid="stSidebar"]{
    background:#FBFBFD;
    border-right:1px solid rgba(0,0,0,0.07);
}

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid rgba(0,0,0,0.06);
    border-radius:20px;
    padding:16px 18px;
    box-shadow:0 6px 18px rgba(0,0,0,0.04);
}

[data-testid="stMetricLabel"]{
    color:#6E6E73;
}

[data-testid="stMetricValue"]{
    color:#1D1D1F;
    letter-spacing:-0.025em;
}

div[role="radiogroup"]{
    background:#ECECF1;
    border-radius:12px;
    padding:3px;
}

div[role="radiogroup"] label:has(input:checked){
    background:white;
    border-radius:9px;
    box-shadow:0 1px 4px rgba(0,0,0,0.10);
}

[data-testid="stExpander"]{
    background:#FFFFFF;
    border:1px solid rgba(0,0,0,0.06);
    border-radius:16px;
}

h1, h2, h3{
    color:#1D1D1F;
    letter-spacing:-0.025em;
}

footer{
    visibility:hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("京粉报")
st.sidebar.caption("BJ Pollenix")

st.sidebar.subheader("公园")
selected_park_name = st.sidebar.radio(
    "选择公园",
    ["北京动物园", "颐和园"],
    horizontal=True,
    label_visibility="collapsed",
)

park = PARKS[selected_park_name]

st.sidebar.caption(
    f"{park.name} · 有效绿地 {park.effective_green_area_ha:.2f} ha"
)

st.sidebar.divider()

# 重点保留：背景花粉浓度滑条
st.sidebar.subheader("当前背景花粉浓度")
st.sidebar.caption("Background pollen density")

pollen_density = st.sidebar.slider(
    "背景花粉浓度（粒/m³）",
    min_value=0,
    max_value=60,
    value=30,
    step=1,
    help=(
        "该值仍直接进入原 BJ Pollenix 的连续型背景花粉算法："
        "midpoint=24，steepness=0.158。"
    ),
)

preview_pollen_score = pollen_score(
    pollen_density
)

st.sidebar.metric(
    "背景花粉评分",
    f"{preview_pollen_score:.1f}/100"
)

st.sidebar.divider()

date = st.sidebar.date_input(
    "日期"
)

with st.sidebar.expander(
    "气象参数",
    expanded=True,
):
    wind_speed = st.slider(
        "风速 (m/s)",
        0.0,
        12.0,
        3.0,
        0.1,
    )

    humidity = st.slider(
        "湿度 (%)",
        0,
        100,
        55,
        1,
    )

    rainfall_48h = st.slider(
        "48h 降雨量 (mm)",
        0.0,
        100.0,
        0.0,
        0.5,
    )

    sunlight_hours = st.slider(
        "日照时长 (h)",
        0.0,
        15.0,
        8.0,
        0.1,
    )

    temperature = st.slider(
        "温度 (°C)",
        -15.0,
        40.0,
        17.0,
        0.5,
    )

    pm25 = st.slider(
        "PM2.5 (μg/m³)",
        0,
        300,
        35,
        1,
    )

with st.sidebar.expander(
    "当前公园数据说明",
    expanded=False,
):
    st.write(park.data_note)
    st.write(park.reliability_note)


# ------------------------------------------------------------
# Model execution
# ------------------------------------------------------------

env = ParkEnvironment(
    date=str(date),
    pollen_density=pollen_density,
    wind_speed=wind_speed,
    humidity=humidity,
    rainfall_48h=rainfall_48h,
    sunlight_hours=sunlight_hours,
    temperature=temperature,
    pm25=pm25,
)

result = calculate_risk(
    env,
    park,
)

risk = result["风险评分"]


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("京粉报")
st.caption(
    f"BJ Pollenix · {park.name} · "
    "基于局部致敏植被、背景花粉与气象条件的实验性风险指数"
)

st.divider()


# ------------------------------------------------------------
# Primary metrics
# ------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "综合风险",
        f"{risk:.1f}/100"
    )

with m2:
    st.metric(
        "风险等级",
        result["风险等级"]
    )

with m3:
    st.metric(
        "背景花粉",
        f"{pollen_density} 粒/m³"
    )

with m4:
    st.metric(
        "天气适宜度",
        f"{result['天气适宜度']:.1f}/100"
    )

st.info(
    result["活动建议"]
)


# ------------------------------------------------------------
# Main charts
# ------------------------------------------------------------

left, right = st.columns(
    [1, 1]
)

with left:
    st.subheader("综合风险")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk,
            number={
                "font": {
                    "size": 38,
                    "color": "#1D1D1F"
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#0071E3",
                    "thickness": 0.24
                },
                "bgcolor": "#ECECF1",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 20],
                        "color": "#F2F2F7"
                    },
                    {
                        "range": [20, 40],
                        "color": "#E8F1FB"
                    },
                    {
                        "range": [40, 60],
                        "color": "#D6E8FA"
                    },
                    {
                        "range": [60, 80],
                        "color": "#BEDCF7"
                    },
                    {
                        "range": [80, 100],
                        "color": "#A7D0F4"
                    }
                ]
            }
        )
    )

    fig_gauge.update_layout(
        height=290,
        margin=dict(
            l=25,
            r=25,
            t=20,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


with right:
    st.subheader("主要过敏树种贡献")

    species_df = pd.DataFrame({
        "树种":
            list(
                result[
                    "物种危险贡献"
                ].keys()
            ),
        "危险贡献 (%)":
            list(
                result[
                    "物种危险贡献"
                ].values()
            )
    })

    fig_species = px.bar(
        species_df,
        x="树种",
        y="危险贡献 (%)",
        text="危险贡献 (%)",
    )

    fig_species.update_traces(
        marker_color="#0071E3",
        textposition="outside",
    )

    fig_species.update_layout(
        height=290,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=None,
            showgrid=False,
        ),
        yaxis=dict(
            title=None,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig_species,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ------------------------------------------------------------
# Secondary metrics — no radar chart to keep the page clean
# ------------------------------------------------------------

st.subheader("植被与环境")

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.metric(
        "标准 I_GZA",
        result["标准 I_GZA"]
    )

with v2:
    st.metric(
        "有效绿地 I_GZA*",
        result["有效绿地 I_GZA*"]
    )

with v3:
    st.metric(
        "植被相对评分",
        result["植被相对评分"]
    )

with v4:
    st.metric(
        "背景花粉评分",
        result["背景花粉评分"]
    )


detail_df = pd.DataFrame(
    result["详细指标"].items(),
    columns=[
        "指标",
        "评分"
    ]
)

fig_detail = px.bar(
    detail_df,
    x="评分",
    y="指标",
    orientation="h",
)

fig_detail.update_traces(
    marker_color="#34A853"
)

fig_detail.update_layout(
    height=330,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        range=[0, 100],
        title=None,
        gridcolor="rgba(0,0,0,0.06)",
    ),
    yaxis=dict(
        title=None,
    ),
    showlegend=False,
)

st.plotly_chart(
    fig_detail,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ------------------------------------------------------------
# Supporting information
# ------------------------------------------------------------

with st.expander(
    "查看原始植被参数",
    expanded=False,
):
    raw_species_df = pd.DataFrame([
        {
            "树种":
                tree.chinese_name,
            "学名":
                tree.scientific_name,
            "株数":
                tree.count,
            "AP":
                tree.allergenic_potential,
            "PE":
                tree.pollen_emission,
            "花期(d)":
                tree.flowering_duration,
            "平均冠幅投影(m²)":
                tree.crown_area,
            "平均树高(m)":
                tree.canopy_height,
            "说明":
                tree.parameter_note,
        }
        for tree
        in park.trees
    ])

    st.dataframe(
        raw_species_df,
        use_container_width=True,
        hide_index=True,
    )


with st.expander(
    "模型方法与局限",
    expanded=False,
):
    st.markdown(
        r"""
### 植被层

采用 Green Zone Allergenicity Index：

\[
I_{GZA}
=
\frac{
\sum(
N_i AP_i PE_i FD_i S_i H_i
)
}{
378 S_T
}
\]

### 有效绿地

- 北京动物园：40 ha
- 颐和园：290.8 × 40% = 116.32 ha

I_GZA* 是 BJ Pollenix 按有效绿地重新归一化的比较指标。

### 背景花粉

背景花粉算法恢复为原模型的连续 logistic 评分：

\[
P =
\frac{100}{
1 + e^{-0.158(x-24)}
}
\]

输入主要限定在 **0–60 粒/m³**。

### 气象层

风速、温度、湿度、降雨、日照用于动态修正。
PM2.5 仅作为实验性的炎症增强因子。

### 数据局限

颐和园的主要树种参数来自文献；
北京动物园的株数来自乔木普查，
其 AP / PE / 花期 / 冠幅 / 树高暂使用颐和园同种或近缘树种参数作为 proxy。
"""
    )


st.caption(
    "BJ Pollenix · Experimental research & education index · "
    "Not a medical diagnosis."
)
