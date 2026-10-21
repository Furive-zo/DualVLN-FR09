import math

import pytest

from dualvln_ros.core import decide, model_to_base_link, parse_command


def test_parse_command():
    assert parse_command('follow') == ('follow', '도로를 따라 주행하라')
    text = '{"command": "safe_stop", "instruction": "오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라"}'
    assert parse_command(text)[0] == 'safe_stop'
    with pytest.raises(ValueError):
        parse_command('safe_stop')       # needs a target description
    with pytest.raises(ValueError):
        parse_command('go')


def test_model_to_base_link():
    # straight ahead 4.5 m: model (0, -4.5, -pi/2) -> base_link (4.5, 0, 0)
    x, y, yaw = model_to_base_link((0.0, -4.5, -math.pi / 2))
    assert (round(x, 6), round(y, 6), round(yaw, 6)) == (4.5, 0.0, 0.0)
    # 0.5 m to the right, heading turned right by 10 deg -> y_left < 0, yaw < 0
    x, y, yaw = model_to_base_link((0.5, -4.0, -math.pi / 2 + math.radians(10)))
    assert y == -0.5 and math.isclose(yaw, -math.radians(10))


def test_decide():
    ok = {'http_status': 200, 'control_decision': 'trajectory', 'trajectory': {'selected': [[0, -1, 0]] * 4}}
    assert decide(ok, 5.0, 1.0) == 'follow'
    assert decide({'http_status': 200, 'control_decision': 'stop'}, 0.1, 1.0) == 'stop'
    assert decide({'http_status': 409, 'control_decision': 'stop'}, 0.1, 1.0) == 'stop'
    timeout = {'http_status': None, 'control_decision': None}
    assert decide(timeout, 0.5, 1.0) == 'hold'
    assert decide(timeout, 1.5, 1.0) == 'stop'
