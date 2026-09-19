import ast
import sys
import types
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]

def load_core(path, name):
    body=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='st.set_page_config':
            break
        if isinstance(n,ast.Import) and any(a.name.startswith(('streamlit','pandas','plotly')) for a in n.names):
            continue
        body.append(n)
    module=types.ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path,monkeypatch):
    monkeypatch.setenv('BJ_POLLENIX_ORIGINAL_CONFIG',str(tmp_path/'original_inputs.json'))

@pytest.fixture
def model():
    return load_core(ROOT/'app.py','exact_second_app')

@pytest.fixture
def original():
    return load_core(ROOT/'tests/fixtures/original_second.py','exact_original_second')

@pytest.fixture
def first():
    return load_core(ROOT/'tests/fixtures/original_first.py','original_park_data')
