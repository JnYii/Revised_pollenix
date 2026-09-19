"""以用户第二份源码为唯一计算基线：逐字、逐函数、逐结果核对。"""
import ast
import calendar
import math
import random
from dataclasses import asdict, fields, replace
from datetime import date, timedelta
import pytest
from conftest import ROOT


def original_block():
    text = (ROOT / 'tests/fixtures/original_second.py').read_text()
    last = max(n.end_lineno for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    return ''.join(text.splitlines(keepends=True)[:last])


def test_entire_second_algorithm_is_byte_for_byte_identical():
    app = (ROOT / 'app.py').read_text()
    block = app.split('# BEGIN ORIGINAL SECOND ALGORITHM\n', 1)[1].split('\n# END ORIGINAL SECOND ALGORITHM', 1)[0]
    assert block == original_block()


def test_no_original_function_or_environment_is_overwritten():
    old = {n.name: n for n in ast.parse(original_block()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = [n for n in ast.parse((ROOT / 'app.py').read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for name, node in old.items():
        matches = [n for n in new if n.name == name]
        assert len(matches) == 1, name
        assert ast.dump(matches[0]) == ast.dump(node), name


def test_default_input_numbers_are_copied_from_original_sidebar(model):
    tree = ast.parse((ROOT / 'tests/fixtures/original_second.py').read_text())
    original_defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            call = node.value
            if isinstance(call, ast.Call) and ast.unparse(call.func) == 'st.sidebar.slider':
                original_defaults[node.targets[0].id] = ast.literal_eval(call.args[3])
    assert model.ORIGINAL_INPUT_DEFAULTS == original_defaults


def test_first_park_inventory_areas_and_igza_are_preserved(model, first):
    assert {k: asdict(v) for k, v in model.PARKS.items()} == {k: asdict(v) for k, v in first.PARKS.items()}
    for name in model.PARKS:
        park = model.PARKS[name]
        for area in (park.total_area_ha, park.effective_green_area_ha):
            assert model.calculate_igza(park, area) == first.calculate_igza(first.PARKS[name], area)


def original_values(model, **changes):
    values = dict(model.ORIGINAL_INPUT_DEFAULTS, date='2026-09-19')
    values.update(changes)
    return values


def compare(model, original, values):
    actual = model.calculate_risk(model.ParkEnvironment(**values))
    expected = original.calculate_risk(original.ParkEnvironment(**values))
    assert actual == expected
    assert 0 <= actual['风险评分'] <= 100
    return actual


@pytest.mark.parametrize('area', [1, 40, 116.32, 120, 150])
def test_default_scenarios_match_every_original_output(model, original, area):
    compare(model, original, original_values(model, green_area=area))


@pytest.mark.parametrize('year', [2024, 2026])
def test_every_date_including_leap_and_month_boundaries_matches_original(model, original, year):
    day = date(year, 1, 1)
    while day.year == year:
        for area in (40, 116.32):
            compare(model, original, original_values(model, date=day.isoformat(), green_area=area))
        day += timedelta(days=1)


def test_1000_random_inputs_match_all_nested_results_exactly(model, original):
    rng = random.Random(20260919)
    for _ in range(1000):
        values = dict(date=(date(2024, 1, 1) + timedelta(days=rng.randrange(1000))).isoformat(),
            pollen_density=rng.uniform(0,55), green_area=rng.uniform(1,150),
            juniper_density=rng.uniform(0,100), poplar_density=rng.uniform(0,100),
            willow_density=rng.uniform(0,100), wind_speed=rng.uniform(0,12),
            humidity=rng.uniform(0,100), rainfall_48h=rng.uniform(0,100),
            sunlight_hours=rng.uniform(0,15), temperature=rng.uniform(-15,40),
            pm25=rng.uniform(0,300), uv_index=rng.uniform(0,15), consecutive_sunny_days=rng.randrange(31))
        compare(model, original, values)


@pytest.mark.parametrize('changes', [
    {'wind_speed':8,'humidity':39}, {'wind_speed':8.00001,'humidity':39},
    {'wind_speed':6,'humidity':81}, {'wind_speed':6.00001,'humidity':81},
    {'rainfall_48h':25}, {'rainfall_48h':25.00001},
    {'pm25':120}, {'pm25':120.00001}, {'uv_index':10}, {'uv_index':10.00001},
    {'rainfall_48h':1,'humidity':71}, {'rainfall_48h':4,'humidity':71},
    {'rainfall_48h':4.00001,'humidity':71}, {'rainfall_48h':1,'humidity':70},
    {'pollen_density':0,'juniper_density':0,'poplar_density':0,'willow_density':0},
])
def test_original_conditional_adjustments_and_boundaries_are_unchanged(model, original, changes):
    compare(model, original, original_values(model, **changes))


def test_density_season_and_wind_helpers_match_original_values(model, original):
    for area in (1, 40, 116.32, 120, 150):
        assert model.estimate_artemisia_density(area) == original.estimate_artemisia_density(area)
        assert model.estimate_grass_density(area) == original.estimate_grass_density(area)
    for month in (1, 3.84, 3.9, 4.1, 6, 8.7, 9, 12):
        for peak, sigma in ((3.84,.95),(4.1,1.1),(3.9,1.1),(9,1.3),(8.7,1.3)):
            assert model.seasonal_factor(month,peak,sigma) == original.seasonal_factor(month,peak,sigma)
    for speed in (0,1.5,2.2,3,7.1,12):
        assert model.artemisia_wind_multiplier(speed) == original.artemisia_wind_multiplier(speed)
        assert model.grass_wind_multiplier(speed) == original.grass_wind_multiplier(speed)


def test_no_igza_or_added_herb_parameters_enter_original_risk(model, original, monkeypatch):
    expected = original.calculate_risk(original.ParkEnvironment(**original_values(model)))
    def fail(*args, **kwargs):
        raise AssertionError('I_GZA must not enter the second model')
    monkeypatch.setattr(model, 'calculate_igza', fail)
    assert model.calculate_risk(model.ParkEnvironment(**original_values(model))) == expected
    assert not hasattr(model, 'HERBS') and not hasattr(model, 'herb_component')
    assert not hasattr(model, 'fractional_month')
