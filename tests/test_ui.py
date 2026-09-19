from pathlib import Path
from datetime import date
import base64
import json
import os
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from conftest import ROOT

LABELS={'pollen_density':'背景花粉浓度（粒/m³）','wind_speed':'风速 (m/s)',
    'humidity':'湿度 (%)','rainfall_48h':'48h 降雨量 (mm)','sunlight_hours':'日照时长 (h)',
    'temperature':'温度 (°C)','pm25':'PM2.5 (μg/m³)','uv_index':'UV指数',
    'consecutive_sunny_days':'连续晴天（天）'}


def by_label(elements,label):
    return next(x for x in elements if x.label==label)


def open_app(park='北京动物园'):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    by_label(app.radio,'选择公园').set_value(park).run()
    assert not app.exception
    return app


def array(data):
    if isinstance(data,dict) and 'bdata' in data:
        return np.frombuffer(base64.b64decode(data['bdata']),dtype=data['dtype']).tolist()
    return data


def check_public_against_original(app,model,original,park,inputs=None):
    assert not app.exception
    values=model.default_park_inputs(model.PARKS[park]) if inputs is None else dict(inputs)
    values.update({k:by_label(app.slider,label).value for k,label in LABELS.items()})
    values['date']=by_label(app.date_input,'日期').value.isoformat()
    expected=original.calculate_risk(original.ParkEnvironment(**values))
    metrics={x.label:x.value for x in app.metric}
    assert metrics['综合风险']==f"{expected['风险评分']:.1f}/100"
    assert metrics['风险等级']==expected['风险等级']
    assert metrics['局部花粉暴露评分']==f"{expected['详细指标']['局部花粉暴露']:.1f}/100"
    assert metrics['天气传播']==f"{expected['详细指标']['天气传播']:.1f}/100"
    assert metrics['植被危险度评分']==str(expected['详细指标']['植被危险度'])
    assert metrics['局部植被增强系数']==str(expected['局部植被增强系数'])
    assert app.info[0].value==expected['活动建议']
    charts=app.get('plotly_chart')
    gauge=json.loads(charts[0].proto.spec)
    assert gauge['data'][0]['value']==expected['风险评分']
    species=json.loads(charts[1].proto.spec)['data'][0]
    assert species['x']==list(expected['过敏源小分'])
    assert array(species['y'])==list(expected['过敏源小分'].values())
    detail=json.loads(charts[2].proto.spec)['data'][0]
    assert detail['y']==list(expected['详细指标'])
    assert array(detail['x'])==list(expected['详细指标'].values())
    return expected


@pytest.mark.parametrize('park',['北京动物园','颐和园'])
def test_public_page_uses_complete_original_result_and_keeps_first_table(model,original,park):
    app=open_app(park)
    check_public_against_original(app,model,original,park)
    first=AppTest.from_file(str(ROOT/'tests/fixtures/original_first.py'),default_timeout=30).run()
    by_label(first.radio,'选择公园').set_value(park).run()
    pd.testing.assert_frame_equal(app.dataframe[0].value,first.dataframe[0].value)
    assert list(app.dataframe[0].value.columns)==[
        '树种','学名','株数','AP','PE','花期(d)','平均冠幅投影(m²)','平均树高(m)','说明']
    for label in ('标准 I_GZA','有效绿地 I_GZA*'):
        assert by_label(app.metric,label).value==by_label(first.metric,label).value
    assert len(app.number_input)==0
    assert [x.label for x in app.slider]==list(LABELS.values())
    for k,label in LABELS.items():
        assert by_label(app.slider,label).value==model.ORIGINAL_INPUT_DEFAULTS[k]


@pytest.mark.parametrize('park',['北京动物园','颐和园'])
def test_date_and_each_weather_input_follow_original_code_exactly(model,original,park):
    app=open_app(park)
    by_label(app.slider,LABELS['pollen_density']).set_value(30)
    by_label(app.slider,LABELS['wind_speed']).set_value(3.0)
    by_label(app.slider,LABELS['humidity']).set_value(55)
    by_label(app.slider,LABELS['temperature']).set_value(17).run()
    results={}
    for d in (date(2024,2,29),date(2026,3,1),date(2026,4,1),date(2026,8,1),date(2026,9,1),date(2026,11,1)):
        by_label(app.date_input,'日期').set_value(d).run()
        results[d.isoformat()]=check_public_against_original(app,model,original,park)
    assert results['2026-04-01']['过敏源小分']!=results['2026-09-01']['过敏源小分']
    assert results['2026-04-01']['风险评分']!=results['2026-09-01']['风险评分']
    by_label(app.date_input,'日期').set_value(date(2026,9,1)).run()
    before=check_public_against_original(app,model,original,park)
    for key,value in [('wind_speed',8.0),('humidity',90),('rainfall_48h',30),
            ('sunlight_hours',2.0),('temperature',32),('pm25',35),('uv_index',2),('consecutive_sunny_days',0)]:
        widget=by_label(app.slider,LABELS[key])
        old=widget.value
        widget.set_value(value).run()
        changed=check_public_against_original(app,model,original,park)
        assert changed['风险评分']!=before['风险评分'],key
        by_label(app.slider,LABELS[key]).set_value(old).run()
        assert check_public_against_original(app,model,original,park)==before


def test_developer_edits_only_original_inputs_and_returns_with_date_weather(model,original):
    app=open_app()
    by_label(app.date_input,'日期').set_value(date(2026,9,1)).run()
    by_label(app.slider,LABELS['rainfall_48h']).set_value(8).run()
    app.checkbox[0].check().run()
    assert app.title[0].value=='开发者模式'
    by_label(app.radio,'输入来源').set_value('填写具体数值').run()
    assert len(app.number_input)==4
    values={'green_area':80.0,'juniper_density':12.0,'poplar_density':7.0,'willow_density':8.0}
    for key,value in values.items():
        next(x for x in app.number_input if x.key==f'original_北京动物园_{key}').set_value(value)
    by_label(app.button,'保存并使用具体数值').click().run()
    assert not app.exception and not app.error and app.success
    by_label(app.button,'返回公众页面').click().run()
    assert by_label(app.date_input,'日期').value==date(2026,9,1)
    assert by_label(app.slider,LABELS['rainfall_48h']).value==8
    check_public_against_original(app,model,original,'北京动物园',values)
    saved=json.loads(Path(os.environ['BJ_POLLENIX_ORIGINAL_CONFIG']).read_text())
    assert saved['parks']['北京动物园']['manual_inputs']==values
    restarted=open_app()
    check_public_against_original(restarted,model,original,'北京动物园',values)
    restarted.checkbox[0].check().run()
    assert by_label(restarted.radio,'输入来源').value=='填写具体数值'
    by_label(restarted.radio,'输入来源').set_value('使用原版默认值').run()
    by_label(restarted.button,'保存并使用原版默认值').click().run()
    by_label(restarted.radio,'输入来源').set_value('填写具体数值').run()
    assert next(x for x in restarted.number_input if x.key=='original_北京动物园_green_area').value==80
    by_label(restarted.radio,'选择公园').set_value('颐和园').run()
    assert by_label(restarted.radio,'输入来源').value=='使用原版默认值'
