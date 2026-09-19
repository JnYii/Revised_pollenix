import json
from pathlib import Path
import pytest


def test_default_park_inputs_change_only_green_area(model):
    for park in model.PARKS.values():
        values = model.default_park_inputs(park)
        assert values == {'green_area':park.effective_green_area_ha,
            'juniper_density':2, 'poplar_density':3, 'willow_density':2}


def test_new_configuration_has_no_automatic_write(model, tmp_path):
    path=tmp_path/'new.json'
    assert model.load_original_settings(path)==({'version':1,'parks':{}},None)
    assert not path.exists()


def test_manual_switch_roundtrip_and_park_independence(model, tmp_path):
    path=tmp_path/'inputs.json'
    zoo,palace=model.PARKS.values()
    values={'green_area':80,'juniper_density':12,'poplar_density':7,'willow_density':8}
    model.save_original_inputs(zoo.name,'manual',values,path)
    settings,error=model.load_original_settings(path)
    assert error is None
    assert model.selected_park_inputs(zoo,settings)==values
    assert model.selected_park_inputs(palace,settings)==model.default_park_inputs(palace)
    model.save_original_inputs(zoo.name,'original',path=path)
    settings,_=model.load_original_settings(path)
    assert model.selected_park_inputs(zoo,settings)==model.default_park_inputs(zoo)
    assert settings['parks'][zoo.name]['manual_inputs']==values
    model.save_original_inputs(zoo.name,'manual',path=path)
    settings,_=model.load_original_settings(path)
    assert model.selected_park_inputs(zoo,settings)==values


@pytest.mark.parametrize('change', [{'green_area':0},{'green_area':151},{'juniper_density':-1},
    {'poplar_density':101},{'willow_density':float('nan')},{'juniper_density':True}])
def test_invalid_input_does_not_replace_existing_file(model,tmp_path,change):
    path=tmp_path/'inputs.json'
    model.save_original_inputs('北京动物园','original',path=path)
    before=path.read_bytes()
    values=dict(model.default_park_inputs(model.PARKS['北京动物园']),**change)
    with pytest.raises(ValueError):
        model.save_original_inputs('北京动物园','manual',values,path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('text',['{bad','null','{}','{"version":1,"parks":{"北京动物园":{"mode":"manual"}}}'])
def test_corrupt_configuration_is_reported_and_not_overwritten(model,tmp_path,text):
    path=tmp_path/'inputs.json'
    path.write_text(text)
    _,error=model.load_original_settings(path)
    assert error
    with pytest.raises(ValueError,match='没有覆盖'):
        model.save_original_inputs('北京动物园','original',path=path)
    assert path.read_text()==text


def test_failed_write_keeps_old_values_and_cleans_up(model,tmp_path,monkeypatch):
    path=tmp_path/'inputs.json'
    model.save_original_inputs('北京动物园','original',path=path)
    before=path.read_bytes()
    def fail(*args):
        raise OSError('test failure')
    monkeypatch.setattr(model.os,'replace',fail)
    with pytest.raises(OSError):
        model.save_original_inputs('北京动物园','manual',model.default_park_inputs(model.PARKS['北京动物园']),path)
    assert path.read_bytes()==before
    assert list(tmp_path.iterdir())==[path]
