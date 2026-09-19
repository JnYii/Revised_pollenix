from dataclasses import dataclass
import math
import json
from datetime import datetime
import streamlit as st


@dataclass
class ParkEnvironment:

    date: str
    # YYYY-MM-DD

    pollen_density: float
    # 粒/m³
    # 北京市区域背景花粉浓度

    green_area: float
    # 公园绿地面积
    # ha

    juniper_density: float
    # 棵/ha

    poplar_density: float
    # 棵/ha

    willow_density: float
    # 棵/ha

    wind_speed: float
    # m/s

    humidity: float
    # %

    rainfall_48h: float
    # mm

    sunlight_hours: float
    # h

    temperature: float
    # 摄氏度

    pm25: float
    # μg/m³

    uv_index: float
    # UVI

    consecutive_sunny_days: int
    # 天


# 基础工具函数

def clamp(
    value,
    minimum=0,
    maximum=100
):

    return max(
        minimum,
        min(value, maximum)
    )


def gaussian_score(
    value,
    optimal,
    sigma,
    amplitude=100
):

    score = amplitude * math.exp(
        -((value - optimal) ** 2) /
        (2 * sigma ** 2)
    )

    return clamp(score)


def logistic_score(
    value,
    midpoint,
    steepness,
    maximum=100
):

    score = maximum / (
        1 + math.exp(
            -steepness *
            (value - midpoint)
        )
    )

    return clamp(score)


def exponential_growth_score(
    value,
    rate=0.5,
    maximum=100
):

    score = maximum * (
        1 - math.exp(
            -rate * value
        )
    )

    return clamp(score)


# 月份距离函数

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


# 季节活跃函数

def seasonal_factor(
    month,
    peak_month,
    sigma,
    baseline=0.0025,
    amplitude=0.94
):

    distance = (
        circular_month_distance(
            month,
            peak_month
        )
    )

    factor = baseline + (
        amplitude *
        math.exp(
            -(distance ** 2) /
            (2 * sigma ** 2)
        )
    )

    return factor


# 城市绿地蒿属估算模型

def estimate_artemisia_density(

    green_area
):
    ecological_density = (
        10 +
        42 *

        (
            1 -
            math.exp(
                -green_area / 18
            )
        )
    )
    management_decay = (
        1 -
        0.18 *
        (
            1 -
            math.exp(
                -green_area / 120
            )
        )
    )
    density = (
        ecological_density *
        management_decay
    )
    return density


# 城市绿地草本植物估算模型

def estimate_grass_density(

    green_area
):
    ecological_density = (
        32 +
        85 *
        (
            1 -
            math.exp(
                -green_area / 26
            )
        )
    )
    management_decay = (
        1 -
        0.10 *
        (
            1 -
            math.exp(
                -green_area / 160
            )
        )
    )
    density = (
        ecological_density *
        management_decay
    )
    return density


# 致敏植物丰度计算

def calculate_allergenic_abundance(
    env,
    month
):

    artemisia_density = (
        estimate_artemisia_density(
            env.green_area
        )
    )

    grass_density = (
        estimate_grass_density(
            env.green_area
        )
    )

    juniper_activity = seasonal_factor(
        month=month,
        peak_month=3.84,
        sigma=0.95
    )

    artemisia_activity = seasonal_factor(
        month=month,
        peak_month=9,
        sigma=1.3
    )

    poplar_activity = seasonal_factor(
        month=month,
        peak_month=4.1,
        sigma=1.1
    )

    willow_activity = seasonal_factor(
        month=month,
        peak_month=3.9,
        sigma=1.1
    )

    grass_activity = seasonal_factor(
        month=month,
        peak_month=8.7,
        sigma=1.3
    )

    weighted_sum = (

    env.juniper_density *
    1.09 *
    juniper_activity +

    artemisia_density *
    1.07 *
    artemisia_activity *
    artemisia_wind_multiplier(
        env.wind_speed
    ) +

    env.poplar_density *
    0.58 *
    poplar_activity +

    env.willow_density *
    0.50 *
    willow_activity +

    grass_density *
    0.52 *
    grass_activity *
    grass_wind_multiplier(
        env.wind_speed
    )
)

    return (
        weighted_sum,
        artemisia_density,
        grass_density
    )


# 局部植被增强函数

def local_density_multiplier(
    allergenic_abundance
):

    multiplier = (
        0.90 +
        0.70 *
        (
            1 -
            math.exp(
                -allergenic_abundance / 185
            )
        )
    )

    return multiplier

# 蒿属风速增强模型

def artemisia_wind_multiplier(
    wind_speed
):

    multiplier = (
        0.38 +
        0.82 *
        math.exp(
            -(
                (wind_speed - 1.5) ** 2
            ) /
            (
                2 * 1.3 ** 2
            )
        )
    )

    return multiplier

# 草本植物风速增强模型

def grass_wind_multiplier(
    wind_speed
):

    multiplier = (
        0.30 +
        0.88 *
        math.exp(
            -(
                (wind_speed - 2.2) ** 2
            ) /
            (
                2 * 1.35 ** 2
            )
        )
    )

    return multiplier

# 花粉暴露评分

def pollen_score(
    local_exposure
):

    return logistic_score(
        value=local_exposure,
        midpoint=24,
        steepness=0.158
    )


# 植被危险度评分

def plant_score(
    allergenic_abundance
):

    return logistic_score(
        value=allergenic_abundance,
        midpoint=185,
        steepness=0.018
    )


# 风速传播模型

def wind_score(
    wind_speed
):

    return gaussian_score(
        value=wind_speed,
        optimal=3.2,
        sigma=1.6,
        amplitude=72
    )


# 湿度影响模型

def humidity_score(
    humidity
):
    return gaussian_score(
        value=humidity,
        optimal=42,
        sigma=20,
        amplitude=77
    )


# 降雨抑制模型

def rainfall_score(
    rainfall_48h
):

    suppression = logistic_score(
        value=rainfall_48h,
        midpoint=6,
        steepness=0.20
    )

    return clamp(
        70 - suppression,
        0,
        100
    )


# 日照影响模型

def sunlight_score(
    hours
):

    return logistic_score(
        value=hours,
        midpoint=6,
        steepness=0.28
    )


# 温度影响模型

def temperature_score(
    temp
):

    return gaussian_score(
        value=temp,
        optimal=22,
        sigma=8
    )


# PM2.5模型

def pm25_score(
    pm25
):

    return logistic_score(
        value=pm25,
        midpoint=95,
        steepness=0.025
    )


# UV模型

def uv_score(
    uv
):

    return gaussian_score(
        value=uv,
        optimal=7,
        sigma=2.6,
        amplitude=52
    )


# 连续晴天模型

def sunny_days_score(
    days
):

    return exponential_growth_score(
        value=days,
        rate=0.14,
        maximum=62
    )


# 风险计算函数

def calculate_risk(
    env: ParkEnvironment
):

    date_obj = datetime.strptime(
        env.date,
        "%Y-%m-%d"
    )

    current_month = (
        date_obj.month +
        (date_obj.day - 1) / 30
    )
    (
        allergenic_abundance,
        artemisia_density,
        grass_density
    ) = calculate_allergenic_abundance(
        env,
        current_month
    )

    density_multiplier = (
        local_density_multiplier(
            allergenic_abundance
        )
    )

    local_exposure = (
        env.pollen_density *
        density_multiplier
    )

    P = pollen_score(
        local_exposure
    )

    A = plant_score(
        allergenic_abundance
    )

    juniper_component = (
        env.juniper_density *
        1.22 *
        seasonal_factor(
            current_month,
            3.84,
            0.95
        )
    )

    artemisia_component = (
        artemisia_density *
        1.05 *
        seasonal_factor(
            current_month,
            9,
            1.3
        )
     *
    artemisia_wind_multiplier(
        env.wind_speed
        )
    )

    poplar_component = (
        env.poplar_density *
        0.50 *
        seasonal_factor(
            current_month,
            4.1,
            1.1
        )
    )

    willow_component = (
        env.willow_density *
        0.42 *
        seasonal_factor(
            current_month,
            3.9,
            1.1
        )
    )

    grass_component = (
    grass_density *
    0.45 *
    seasonal_factor(
        current_month,
        8.7,
        1.3
    ) *
    grass_wind_multiplier(
        env.wind_speed
    )
)

    total_component = max(
        juniper_component +
        artemisia_component +
        poplar_component +
        willow_component +
        grass_component,
        1e-6
    )

    pollen_load = (
        0.85 * P +
        0.15 * A
    )

    allergen_subscores = {

        "刺柏": round(
            pollen_load *
            (
                juniper_component /
                total_component
            ),
            2
        ),

        "蒿属": round(
            pollen_load *
            (
                artemisia_component /
                total_component
            ),
            2
        ),

        "杨树": round(
            pollen_load *
            (
                poplar_component /
                total_component
            ),
            2
        ),

        "柳树": round(
            pollen_load *
            (
                willow_component /
                total_component
            ),
            2
        ),

        "草本植物": round(
            pollen_load *
            (
                grass_component /
                total_component
            ),
            2
        )
    }

    W = wind_score(
        env.wind_speed
    )

    S = sunlight_score(
        env.sunlight_hours
    )

    T = temperature_score(
        env.temperature
    )

    weather_transport = (
        0.45 * W +
        0.25 * S +
        0.30 * T
    )

    H = humidity_score(
        env.humidity
    )

    C = sunny_days_score(
        env.consecutive_sunny_days
    )

    R = rainfall_score(
        env.rainfall_48h
    )

    atmospheric_retention = (
        0.38 * H +
        0.14 * C +
        0.48 * R
    )

    M = pm25_score(
        env.pm25
    )

    inflammatory_multiplier = (
        1 +
        0.1 *
        (M / 100)
    )

    U = uv_score(
        env.uv_index
    )

    bioactivity_factor = (
        0.52 * T +
        0.14 * U +
        0.34 * S
    )

    raw_exposure = (
    pollen_load
    *
    (
        0.72 +
        0.28 *
        weather_transport / 100
    )
    *
    (
        0.82 +
        0.18 *
        atmospheric_retention / 100
    )
)

    base_exposure = 100 * (
        1 - math.exp(
            -raw_exposure / 56
        )
    )

    biologically_active_exposure = (
        0.94 * base_exposure +
        0.06 * bioactivity_factor
    )

    inflammatory_risk = (
        biologically_active_exposure *
        inflammatory_multiplier
    )

    if (
        env.wind_speed > 8 and
        env.humidity < 40
    ):
        inflammatory_risk += 2.5

    if (
        env.humidity > 80 and
        env.wind_speed > 6
    ):
        inflammatory_risk += 1.5

    if env.rainfall_48h > 25:
        inflammatory_risk -= 7

    if env.pm25 > 120:
        inflammatory_risk += 2.5

    if env.uv_index > 10:
        inflammatory_risk += 1

    if (
        1 <= env.rainfall_48h <= 4 and
        env.humidity > 70
    ):
        inflammatory_risk += 3

    total_risk = clamp(
        inflammatory_risk
    )

    if total_risk < 15:

        level = "安全"

    elif total_risk < 35:

        level = "轻度风险"

    elif total_risk < 55:

        level = "中度风险"

    elif total_risk < 75:

        level = "高风险"

    else:

        level = "极高风险"

    if total_risk >= 80:

        recommendation = (
            "不建议过敏患者进入园区"
        )

    elif total_risk >= 60:

        recommendation = (
            "建议佩戴N95口罩并减少停留"
        )

    elif total_risk >= 40:

        recommendation = (
            "敏感人群建议谨慎活动"
        )

    else:

        recommendation = (
            "可正常活动"
        )

    factors = {

        "局部花粉暴露": P,

        "植被危险度": A,

        "风速扩散": W,

        "湿度影响": H,

        "降雨情况": R,

        "日照": S,

        "温度": T,

        "PM2.5": M,

        "UV": U,

        "连续晴天": C
    }

    sorted_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_risks = sorted_factors[:3]

    return {

        "风险评分":
            round(total_risk, 2),

        "风险等级":
            level,

        "活动建议":
            recommendation,

        "估算蒿属密度":
            round(
                artemisia_density,
                2
            ),

        "估算草本密度":
            round(
                grass_density,
                2
            ),

        "局部植被增强系数":
            round(
                density_multiplier,
                3
            ),

        "局部真实花粉暴露":
            round(
                local_exposure,
                2
            ),

        "致敏植物加权丰度":
            round(
                allergenic_abundance,
                2
            ),

        "过敏源小分":
            allergen_subscores,

        "主要风险因素":
            top_risks,

        "详细指标": {

            "局部花粉暴露":
                round(P, 2),

            "植被危险度":
                round(A, 2),

            "天气传播":
                round(
                    weather_transport,
                    2
                ),

            "空气滞留":
                round(
                    atmospheric_retention,
                    2
                ),

            "花粉生物活性":
                round(
                    bioactivity_factor,
                    2
                ),

            "基础暴露":
                round(
                    base_exposure,
                    2
                ),

            "生物活性暴露":
                round(
                    biologically_active_exposure,
                    2
                ),

            "炎症增强系数":
                round(
                    inflammatory_multiplier,
                    3
                )
        }
    }

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

st.set_page_config(
    page_title="京粉报",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: "Inter", "Noto Sans SC", sans-serif;
}

.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1{
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #183A37;
}

h2{
    font-size: 1.45rem !important;
    color: #183A37;
}

h3{
    font-size: 1.1rem !important;
    color: #183A37;
}

p, div{
    font-size: 0.95rem;
}

.small-en{
    color:#7B8A8B;
    font-size:0.78rem;
    margin-top:-5px;
    margin-bottom:12px;
}

[data-testid="stMetric"]{
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    border: 1px solid #E7ECEF;
}

[data-testid="stMetricLabel"]{
    font-size: 0.92rem;
}

[data-testid="stMetricValue"]{
    font-size: 1.45rem;
    color: #2D6A73;
}

.stProgress > div > div > div > div{
    background-color: #2D6A73;
}

section[data-testid="stSidebar"]{
    background-color: #F7FAFB;
    border-right: 1px solid #E4ECEE;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
display:flex;
flex-direction:column;
align-items:center;
justify-content:center;
margin-top:10px;
margin-bottom:28px;
">

<div style="
font-size:72px;
font-weight:800;
line-height:1;
color:#2B2D42;
">
京粉报
</div>

<div style="
font-size:20px;
font-weight:500;
color:#6B7280;
margin-top:14px;
letter-spacing:0.5px;
">
BJ Pollenix
</div>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
## 环境参数
<div class='small-en'>
Environmental Parameters
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
### 日期
<div class='small-en'>
Date
</div>
""", unsafe_allow_html=True)

date = st.sidebar.date_input(
    "",
    key="date_input"
)

st.sidebar.markdown("""
### 背景花粉浓度 (粒/m³)
<div class='small-en'>
Background Pollen Density
</div>
""", unsafe_allow_html=True)

pollen_density = st.sidebar.slider(
    "",
    0,
    55,
    46,
    key="pollen_density"
)

st.sidebar.markdown("""
### 公园绿地面积 (ha)
<div class='small-en'>
Urban Green Area
</div>
""", unsafe_allow_html=True)

green_area = st.sidebar.slider(
    "",
    1,
    150,
    120,
    key="green_area"
)

st.sidebar.markdown("""
### 刺柏密度
<div class='small-en'>
Juniper Density
</div>
""", unsafe_allow_html=True)

juniper_density = st.sidebar.slider(
    "",
    0,
    100,
    2,
    key="juniper_density"
)

st.sidebar.markdown("""
### 杨树密度
<div class='small-en'>
Poplar Density
</div>
""", unsafe_allow_html=True)

poplar_density = st.sidebar.slider(
    "",
    0,
    100,
    3,
    key="poplar_density"
)

st.sidebar.markdown("""
### 柳树密度
<div class='small-en'>
Willow Density
</div>
""", unsafe_allow_html=True)

willow_density = st.sidebar.slider(
    "",
    0,
    100,
    2,
    key="willow_density"
)

st.sidebar.markdown("""
### 风速 (m/s)
<div class='small-en'>
Wind Speed
</div>
""", unsafe_allow_html=True)

wind_speed = st.sidebar.slider(
    "",
    0.0,
    12.0,
    7.1,
    key="wind_speed"
)

st.sidebar.markdown("""
### 湿度 (%)
<div class='small-en'>
Humidity
</div>
""", unsafe_allow_html=True)

humidity = st.sidebar.slider(
    "",
    0,
    100,
    20,
    key="humidity"
)

st.sidebar.markdown("""
### 48h降雨量 (mm)
<div class='small-en'>
Rainfall in Last 48 Hours
</div>
""", unsafe_allow_html=True)

rainfall_48h = st.sidebar.slider(
    "",
    0,
    100,
    0,
    key="rainfall_48h"
)

st.sidebar.markdown("""
### 日照时长 (h)
<div class='small-en'>
Sunlight Duration
</div>
""", unsafe_allow_html=True)

sunlight_hours = st.sidebar.slider(
    "",
    0.0,
    15.0,
    11.2,
    key="sunlight_hours"
)

st.sidebar.markdown("""
### 温度 (°C)
<div class='small-en'>
Temperature
</div>
""", unsafe_allow_html=True)

temperature = st.sidebar.slider(
    "",
    -15,
    40,
    30,
    key="temperature"
)

st.sidebar.markdown("""
### PM2.5
<div class='small-en'>
Fine Particulate Matter
</div>
""", unsafe_allow_html=True)

pm25 = st.sidebar.slider(
    "",
    0,
    300,
    148,
    key="pm25"
)

st.sidebar.markdown("""
### UV指数
<div class='small-en'>
UV Index
</div>
""", unsafe_allow_html=True)

uv_index = st.sidebar.slider(
    "",
    0,
    15,
    9,
    key="uv_index"
)

st.sidebar.markdown("""
### 连续晴天
<div class='small-en'>
Consecutive Sunny Days
</div>
""", unsafe_allow_html=True)

consecutive_sunny_days = st.sidebar.slider(
    "",
    0,
    30,
    11,
    key="sunny_days"
)

env = ParkEnvironment(

    date=str(date),

    pollen_density=pollen_density,

    green_area=green_area,

    juniper_density=juniper_density,

    poplar_density=poplar_density,

    willow_density=willow_density,

    wind_speed=wind_speed,

    humidity=humidity,

    rainfall_48h=rainfall_48h,

    sunlight_hours=sunlight_hours,

    temperature=temperature,

    pm25=pm25,

    uv_index=uv_index,

    consecutive_sunny_days=consecutive_sunny_days
)

result = calculate_risk(env)

risk = result["风险评分"]

risk_color = "#DCEFEB"

if risk >= 25:
    risk_color = "#B7DAD4"

if risk >= 50:
    risk_color = "#6EA8A1"

if risk >= 75:
    risk_color = "#2D6A73"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "综合风险评分",
        f"{risk}/100"
    )
    st.markdown(
        "<div class='small-en'>Overall Risk Score</div>",
        unsafe_allow_html=True
    )

with col2:
    st.metric(
        "风险等级",
        result["风险等级"]
    )
    st.markdown(
        "<div class='small-en'>Risk Level</div>",
        unsafe_allow_html=True
    )

with col3:
    st.metric(
        "活动建议",
        result["活动建议"]
    )
    st.markdown(
        "<div class='small-en'>Outdoor Recommendation</div>",
        unsafe_allow_html=True
    )

st.markdown("""
## 综合风险指数
<div class='small-en'>
Overall Allergy Risk Index
</div>
""", unsafe_allow_html=True)

st.progress(
    min(int(risk), 100)
)

st.markdown(
    f"""
<div style="
background:{risk_color};
padding:14px;
border-radius:18px;
text-align:center;
color:white;
font-size:22px;
font-weight:700;
margin-top:12px;
box-shadow:0 4px 14px rgba(0,0,0,0.06);
">
当前综合风险：{risk}/100
</div>
""",
    unsafe_allow_html=True
)

left, right = st.columns([1, 1])

with left:

    st.markdown("""
    ### 综合风险仪表盘
    <div class='small-en'>
    Overall Risk Gauge
    </div>
    """, unsafe_allow_html=True)

    fig_gauge = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=risk,

            number={
                'font': {'size': 34}
            },

            title={
                'text': "Risk Index",
                'font': {'size': 18}
            },

            gauge={

                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color': "#2D6A73",
                    'thickness': 0.35
                },

                'bgcolor': "#EDF3F4",

                'steps': [

                    {
                        'range': [0, 25],
                        'color': "#DCEFEB"
                    },

                    {
                        'range': [25, 50],
                        'color': "#B7DAD4"
                    },

                    {
                        'range': [50, 75],
                        'color': "#6EA8A1"
                    },

                    {
                        'range': [75, 100],
                        'color': "#2D6A73"
                    }
                ]
            }
        )
    )

    fig_gauge.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10)
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

with right:

    st.markdown("""
    ### 主要风险因素
    <div class='small-en'>
    Primary Risk Factors
    </div>
    """, unsafe_allow_html=True)

    factor_df = pd.DataFrame(

        result["主要风险因素"],

        columns=[
            "因素",
            "评分"
        ]
    )

    fig_pie = px.pie(

        factor_df,

        names="因素",

        values="评分",

        hole=0.58,

        color_discrete_sequence=[
            "#DCEFEB",
            "#B7DAD4",
            "#6EA8A1"
        ]
    )

    fig_pie.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(size=12)
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

left2, right2 = st.columns([1, 1])

with left2:

    st.markdown("""
    ### 过敏源贡献分析
    <div class='small-en'>
    Allergen Contribution Analysis
    </div>
    """, unsafe_allow_html=True)

    allergen_df = pd.DataFrame({

        "过敏源":
        list(result["过敏源小分"].keys()),

        "贡献值":
        list(result["过敏源小分"].values())
    })

    fig_bar = px.bar(

        allergen_df,

        x="过敏源",

        y="贡献值",

        text="贡献值",

        color="贡献值",

        color_continuous_scale=[
            "#DCEFEB",
            "#2D6A73"
        ]
    )

    fig_bar.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
        font=dict(size=12)
    )

    fig_bar.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

with right2:

    st.markdown("""
    ### 环境风险雷达图
    <div class='small-en'>
    Environmental Risk Radar
    </div>
    """, unsafe_allow_html=True)

    radar_categories = [

        "花粉暴露",
        "风速",
        "湿度",
        "PM2.5",
        "UV",
        "温度",
        "连续晴天"
    ]

    radar_values = [

        result["详细指标"]["局部花粉暴露"],

        wind_speed * 8,

        humidity,

        pm25 / 3,

        uv_index * 6,

        temperature * 2,

        consecutive_sunny_days * 3
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(

        go.Scatterpolar(

            r=radar_values,

            theta=radar_categories,

            fill='toself',

            line=dict(
                color="#2D6A73"
            ),

            fillcolor="rgba(45,106,115,0.35)"
        )
    )

    fig_radar.update_layout(

        polar=dict(

            radialaxis=dict(

                visible=True,

                range=[0, 100]
            )
        ),

        showlegend=False,

        height=320,

        margin=dict(l=20, r=20, t=20, b=10),

        font=dict(size=11)
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )

st.markdown("""
### 详细指标数据
<div class='small-en'>
Detailed Environmental Indicators
</div>
""", unsafe_allow_html=True)

st.dataframe(
    pd.DataFrame(
        result["详细指标"].items(),
        columns=["指标", "数值"]
    ),
    use_container_width=True
)