# 京粉报：第二份算法，仅收窄草本季节曲线；蒿属宽度按文献季长折算，其余草本宽度保留0.8。
# BEGIN SECOND ALGORITHM
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
        sigma=(47 + 56) / 2 / 4 / 30
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
        sigma=0.8
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
            (47 + 56) / 2 / 4 / 30
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
        0.8
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

# END SECOND ALGORITHM

from pathlib import Path
import os
import tempfile
from datetime import date as Date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 第一份公园资料；仅用于展示，未混入第二份评分公式。
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


# 以下只保存与展示原算法输入；不修改原calculate_risk或任何模型函数。
ORIGINAL_INPUT_DEFAULTS = {
    'green_area': 120.0, 'juniper_density': 2.0, 'poplar_density': 3.0, 'willow_density': 2.0,
    'pollen_density': 46, 'wind_speed': 7.1, 'humidity': 20, 'rainfall_48h': 0,
    'sunlight_hours': 11.2, 'temperature': 30, 'pm25': 148, 'uv_index': 9,
    'consecutive_sunny_days': 11,
}
PARK_INPUT_FIELDS = ('green_area', 'juniper_density', 'poplar_density', 'willow_density')


def original_settings_path():
    return Path(os.environ.get('BJ_POLLENIX_ORIGINAL_CONFIG',
        str(Path(__file__).with_name('original_inputs.json'))))


def default_park_inputs(park):
    values = {k: ORIGINAL_INPUT_DEFAULTS[k] for k in PARK_INPUT_FIELDS}
    values['green_area'] = park.effective_green_area_ha
    return values


def validate_original_settings(settings):
    if not isinstance(settings, dict) or settings.get('version') != 1 or not isinstance(settings.get('parks'), dict):
        raise ValueError('原模型输入文件格式不正确')
    for name, row in settings['parks'].items():
        if name not in PARKS or not isinstance(row, dict) or row.get('mode') not in ('original', 'manual'):
            raise ValueError('无效的公园或输入模式')
        values = row.get('manual_inputs')
        if values is None:
            if row['mode'] == 'manual':
                raise ValueError('缺少已填写的原模型输入')
            continue
        if not isinstance(values, dict) or set(values) != set(PARK_INPUT_FIELDS):
            raise ValueError('须填写有效绿地面积和三项原模型树木密度')
        for field, value in values.items():
            low, high = (1, 150) if field == 'green_area' else (0, 100)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{field}超出原输入范围或不是有限数值')
    return settings


def load_original_settings(path=None):
    path = Path(path) if path is not None else original_settings_path()
    if not path.exists():
        return {'version': 1, 'parks': {}}, None
    try:
        return validate_original_settings(json.loads(path.read_text(encoding='utf-8'))), None
    except (ValueError, OSError) as error:
        return {'version': 1, 'parks': {}}, str(error)


def save_original_inputs(park_name, mode, values=None, path=None):
    path = Path(path) if path is not None else original_settings_path()
    settings, error = load_original_settings(path)
    if error:
        raise ValueError('原输入文件有误，没有覆盖：' + error)
    row = dict(settings['parks'].get(park_name, {}))
    row['mode'] = mode
    if values is not None:
        row['manual_inputs'] = dict(values)
    settings['parks'][park_name] = row
    validate_original_settings(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                prefix='.original_inputs-', suffix='.tmp', delete=False) as handle:
            temp_name = handle.name
            json.dump(settings, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and Path(temp_name).exists():
            Path(temp_name).unlink()


def selected_park_inputs(park, settings):
    row = settings['parks'].get(park.name, {})
    if row.get('mode') == 'manual':
        return dict(row['manual_inputs'])
    return default_park_inputs(park)


def leave_developer_page():
    st.session_state['developer_mode'] = False


def render_original_developer_page(park, settings):
    st.title('开发者模式')
    st.caption(f'{park.name} · 第二份原模型输入')
    st.button('返回公众页面', on_click=leave_developer_page)
    row = settings['parks'].get(park.name, {'mode': 'original'})
    message = st.session_state.pop('original_saved', None)
    if message:
        st.success(message)
    st.write('蒿属和草本密度由原代码根据绿地面积计算。这里只填写原模型已有的输入，不调整算法系数。')
    mode = st.radio('输入来源', ['使用原版默认值', '填写具体数值'],
        index=0 if row['mode'] == 'original' else 1, key=f'original_mode_{park.name}', horizontal=True)
    labels = {'green_area': '有效绿地面积（ha）', 'juniper_density': '刺柏密度（棵/ha）',
        'poplar_density': '杨树密度（棵/ha）', 'willow_density': '柳树密度（棵/ha）'}
    if mode == '使用原版默认值':
        values = default_park_inputs(park)
        st.dataframe(pd.DataFrame([{'参数': labels[k], '数值': values[k]} for k in PARK_INPUT_FIELDS]),
            hide_index=True, use_container_width=True)
        st.caption('绿地面积沿用第一份公园资料；三项密度初值2、3、2沿用第二份原码，不自动把其他树种并入刺柏。')
        if st.button('保存并使用原版默认值', type='primary'):
            try:
                save_original_inputs(park.name, 'original')
            except (ValueError, OSError) as error:
                st.error(f'没有保存成功：{error}')
            else:
                st.session_state['original_saved'] = '已使用原版默认值；之前填写的具体数值仍保留。'
                st.rerun()
    else:
        previous = row.get('manual_inputs', default_park_inputs(park))
        with st.form(f'original_inputs_{park.name}'):
            values = {}
            for key in PARK_INPUT_FIELDS:
                low, high = (1.0, 150.0) if key == 'green_area' else (0.0, 100.0)
                values[key] = st.number_input(labels[key], low, high, float(previous[key]),
                    .01, format='%.2f', key=f'original_{park.name}_{key}')
            submitted = st.form_submit_button('保存并使用具体数值', type='primary')
        if submitted:
            try:
                save_original_inputs(park.name, 'manual', values)
            except (ValueError, OSError) as error:
                st.error(f'没有保存成功：{error}')
            else:
                st.session_state['original_saved'] = '具体数值已保存，将直接进入原模型。'
                st.rerun()
    current = selected_park_inputs(park, settings)
    st.write(f"当前原式估算：蒿属密度 {estimate_artemisia_density(current['green_area']):.2f}；"
        f"草本密度 {estimate_grass_density(current['green_area']):.2f}。")
    st.caption('第二份原码没有草本覆盖率、株高、独立藜科或手填草本密度接口。本版不把这些量换算进原公式；此前herb_settings.json中的记录保留在原文件中。')


st.set_page_config(page_title="京粉报 · BJ Pollenix", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
}
.stApp { background: #F7F7F9; color: #25272B; }
.block-container { max-width: 1240px; padding-top: 4rem; padding-bottom: 2.5rem; }
h1 { font-size: 26px !important; font-weight: 700 !important; letter-spacing: -.025em; }
h2 { font-size: 18px !important; font-weight: 650 !important; }
h3 { font-size: 16px !important; font-weight: 650 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p { font-size: 14px; line-height: 1.55; }
[data-testid="stCaptionContainer"] p { font-size: 12px; color: #737780; }
section[data-testid="stSidebar"] { background: #FAFAFC; border-right: 1px solid #E3E4E8; }
section[data-testid="stSidebar"] h1 { font-size: 20px !important; }
section[data-testid="stSidebar"] h3 { font-size: 16px !important; }
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E3E4E8; border-radius: 16px;
    padding: 11px 15px; box-shadow: none;
}
[data-testid="stMetricLabel"] p { color: #747983; font-size: 13px !important; line-height: 1.4; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: #25272B; font-size: 20px !important; line-height: 1.28 !important;
    font-weight: 500 !important; letter-spacing: -.025em;
}
div[role="radiogroup"] { background: #F0F0F3; border-radius: 12px; padding: 3px; }
div[role="radiogroup"] label:has(input:checked) {
    background: white; border-radius: 9px; box-shadow: 0 2px 5px rgba(0,0,0,.07);
}
[data-testid="stExpander"] { background: transparent; border: 1px solid #D9DBE1; border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stAlert"] p { font-size: 14px; }
.pollenix-hero {
    background: #FFFFFF; border: 1px solid #E3E4E8; border-radius: 18px;
    padding: 15px 22px 16px; margin-bottom: 12px;
}
.pollenix-hero .eyebrow { color: #737780; font-size: 10px; line-height: 1.2; font-weight: 700; letter-spacing: .045em; }
.pollenix-hero h1 { margin: 4px 0 6px; padding: 0; line-height: 1.25; }
.pollenix-hero p { color: #737780; margin: 0 0 11px; font-size: 13px; line-height: 1.5; }
.pollenix-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.pollenix-chip {
    display: inline-block; padding: 5px 10px; border-radius: 999px;
    background: #F0F0F3; color: #555960; font-size: 11px; line-height: 1.4; font-weight: 600;
}
.pollenix-chip.park { background: #E8F6F2; color: #078B74; }
.pollenix-panel-title { font-size: 14px; font-weight: 650; line-height: 1.5; }
.pollenix-panel-subtitle { font-size: 11px; line-height: 1.5; color: #737780; margin-top: 3px; }
footer { visibility: hidden; }
@media (min-width: 768px) {
    section[data-testid="stSidebar"] { min-width: 280px; max-width: 280px; }
    .block-container { padding-left: 3rem; padding-right: 3rem; }
}
@media (max-width: 767px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 4rem; }
    .pollenix-hero { padding: 17px 18px; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-size: 20px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title('京粉报')
st.sidebar.caption('BJ Pollenix')
st.sidebar.subheader('公园')
selected_park_name = st.sidebar.radio('选择公园', ['北京动物园', '颐和园'],
    horizontal=True, label_visibility='collapsed')
park = PARKS[selected_park_name]
settings, settings_error = load_original_settings()
park_inputs = selected_park_inputs(park, settings)
st.sidebar.caption(f"{park.name} · 有效绿地 {park_inputs['green_area']:.2f} ha")
developer_mode = st.sidebar.checkbox('开发者模式', key='developer_mode')
if settings_error:
    st.sidebar.error('已保存的输入读取失败，当前使用原版默认值；请在开发者模式检查。')
st.sidebar.divider()
st.sidebar.subheader('当前背景花粉浓度')
st.sidebar.caption('Background pollen density')
pollen_density = st.sidebar.slider('背景花粉浓度（粒/m³）', 0, 55,
    ORIGINAL_INPUT_DEFAULTS['pollen_density'], 1)
st.sidebar.metric('背景花粉评分', f'{pollen_score(pollen_density):.1f}/100')
st.sidebar.divider()
date = st.sidebar.date_input('日期')
with st.sidebar.expander('气象参数', expanded=True):
    wind_speed = st.slider('风速 (m/s)', 0.0, 12.0, ORIGINAL_INPUT_DEFAULTS['wind_speed'], .1)
    humidity = st.slider('湿度 (%)', 0, 100, ORIGINAL_INPUT_DEFAULTS['humidity'], 1)
    rainfall_48h = st.slider('48h 降雨量 (mm)', 0, 100, ORIGINAL_INPUT_DEFAULTS['rainfall_48h'], 1)
    sunlight_hours = st.slider('日照时长 (h)', 0.0, 15.0, ORIGINAL_INPUT_DEFAULTS['sunlight_hours'], .1)
    temperature = st.slider('温度 (°C)', -15, 40, ORIGINAL_INPUT_DEFAULTS['temperature'], 1)
    pm25 = st.slider('PM2.5 (μg/m³)', 0, 300, ORIGINAL_INPUT_DEFAULTS['pm25'], 1)
    uv_index = st.slider('UV指数', 0, 15, ORIGINAL_INPUT_DEFAULTS['uv_index'], 1)
    consecutive_sunny_days = st.slider('连续晴天（天）', 0, 30,
        ORIGINAL_INPUT_DEFAULTS['consecutive_sunny_days'], 1)
with st.sidebar.expander('当前公园数据说明', expanded=False):
    st.write(park.data_note)
    st.write(park.reliability_note)
if developer_mode:
    if settings_error:
        st.error(f'保存的输入未能读取：{settings_error}')
    render_original_developer_page(park, settings)
    st.stop()

env = ParkEnvironment(date=str(date), pollen_density=pollen_density,
    green_area=park_inputs['green_area'], juniper_density=park_inputs['juniper_density'],
    poplar_density=park_inputs['poplar_density'], willow_density=park_inputs['willow_density'],
    wind_speed=wind_speed, humidity=humidity, rainfall_48h=rainfall_48h,
    sunlight_hours=sunlight_hours, temperature=temperature, pm25=pm25,
    uv_index=uv_index, consecutive_sunny_days=consecutive_sunny_days)
# 原函数直接生成全部结果；没有输出修正、放大或保底。
result = calculate_risk(env)
risk = result['风险评分']

st.markdown(f'''
<div class="pollenix-hero">
    <div class="eyebrow">BJ POLLENIX · PARK ALLERGY RISK</div>
    <h1>京粉报</h1>
    <p>基于局部致敏植被、背景花粉浓度和气象条件，为公园场景生成动态实验性花粉过敏风险指数。</p>
    <div class="pollenix-chips">
        <span class="pollenix-chip park">{park.name}</span>
        <span class="pollenix-chip">有效绿地 {park_inputs['green_area']:.1f} ha</span>
        <span class="pollenix-chip">背景花粉 {pollen_density} 粒/m³</span>
        <span class="pollenix-chip">日期 {date}</span>
    </div>
</div>
''', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric('综合风险', f'{risk:.1f}/100')
with m2:
    st.metric('风险等级', result['风险等级'])
with m3:
    st.metric('局部花粉暴露评分', f"{result['详细指标']['局部花粉暴露']:.1f}/100")
with m4:
    st.metric('天气传播', f"{result['详细指标']['天气传播']:.1f}/100")
st.info(result['活动建议'])

left, right = st.columns([1.1, 1])
with left:
    with st.container(border=True):
        st.markdown('<div class="pollenix-panel-title">综合风险指数</div>'
            '<div class="pollenix-panel-subtitle">Overall risk index</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=risk,
            number={'font': {'size': 34, 'color': '#25272B'}}, gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 11, 'color': '#737780'}},
                'bar': {'color': '#0BA58B', 'thickness': .24},
                'bgcolor': '#F0F2F1', 'borderwidth': 0,
                'steps': [
                    {'range': [0, 15], 'color': '#F8FAF9'},
                    {'range': [15, 35], 'color': '#F0F7F4'},
                    {'range': [35, 55], 'color': '#E5F3EE'},
                    {'range': [55, 75], 'color': '#D8EDE6'},
                    {'range': [75, 100], 'color': '#C4E7DC'},
                ],
            }))
        fig_gauge.update_layout(height=250, margin=dict(l=24, r=24, t=12, b=12),
            paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
with right:
    with st.container(border=True):
        st.markdown('<div class="pollenix-panel-title">植被危险度</div>'
            '<div class="pollenix-panel-subtitle">Vegetation allergenicity</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1:
            st.metric('标准 I_GZA', round(calculate_igza(park, park.total_area_ha), 4))
        with v2:
            st.metric('有效绿地 I_GZA*', round(calculate_igza(park, park.effective_green_area_ha), 4))
        v3, v4 = st.columns(2)
        with v3:
            st.metric('植被危险度评分', result['详细指标']['植被危险度'])
        with v4:
            st.metric('局部植被增强系数', result['局部植被增强系数'])
        st.caption('I_GZA两项保留为公园资料展示；综合风险按第二份原模型计算。')
with st.container(border=True):
    st.markdown('**过敏源贡献分析**')
    species_df = pd.DataFrame({'过敏源': list(result['过敏源小分']),
        '贡献值': list(result['过敏源小分'].values())})
    fig_species = px.bar(species_df, x='过敏源', y='贡献值', text='贡献值')
    fig_species.update_traces(marker_color='#0BA58B', textposition='outside', textfont_size=12)
    fig_species.update_layout(height=270, margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=12, color='#666B75'),
        xaxis=dict(title=None, showgrid=False), yaxis=dict(title=None, gridcolor='rgba(0,0,0,.06)', zeroline=False),
        showlegend=False)
    st.plotly_chart(fig_species, use_container_width=True, config={'displayModeBar': False})
with st.container(border=True):
    st.markdown('**植被与环境**')
    # 展示原返回值，不把原来的局部暴露或丰度重新换算成第一份的指标。
    detail_df = pd.DataFrame(result['详细指标'].items(), columns=['指标', '评分'])
    fig_detail = px.bar(detail_df, x='评分', y='指标', orientation='h')
    fig_detail.update_traces(marker_color='#0BA58B')
    fig_detail.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=12, color='#666B75'),
        xaxis=dict(range=[0,100], title=None, gridcolor='rgba(0,0,0,.06)'), yaxis=dict(title=None), showlegend=False)
    st.plotly_chart(fig_detail, use_container_width=True, config={'displayModeBar': False})


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


with st.expander("模型方法与局限", expanded=False):
    st.write("综合评分沿用第二份代码：逐种季节活跃度与风速影响计算植物丰度，"
        "再计算局部花粉暴露、天气传播、空气滞留、生物活性和PM2.5影响。")
    st.write("蒿属与草本密度由原来的绿地面积函数估算；本版仅收窄这两项的季节曲线，"
        "降低远离花期时的活跃度，其余参数保留。没有独立的藜科参数。")
    st.write("I_GZA与原始植被表保留为公园资料展示，不参与这套总分计算。"
        "密度初值沿用原第二份输入，可在开发者模式填写掌握的数值。")
st.caption("BJ Pollenix · Experimental research & education index · Not a medical diagnosis.")
