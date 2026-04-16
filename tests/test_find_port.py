#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.scripts import lerobot_find_port


def test_find_available_ports_prefers_pyserial(monkeypatch):
    comports = [
        Mock(device="/dev/ttyACM1"),
        Mock(device="/dev/ttyACM0"),
        Mock(device="/dev/ttyACM0"),
    ]
    monkeypatch.setattr(lerobot_find_port.list_ports, "comports", lambda: comports)

    assert lerobot_find_port.find_available_ports() == ["/dev/ttyACM0", "/dev/ttyACM1"]


def test_find_available_ports_uses_usb_serial_fallback_on_unix(monkeypatch):
    monkeypatch.setattr(lerobot_find_port.list_ports, "comports", lambda: [])
    monkeypatch.setattr(lerobot_find_port.os, "name", "posix")

    class FakeDevDir:
        def glob(self, pattern: str) -> list[Path]:
            matches = {
                "ttyACM*": [Path("/dev/ttyACM0")],
                "ttyUSB*": [Path("/dev/ttyUSB0")],
                "ttyAMA*": [],
                "tty.usb*": [Path("/dev/tty.usbmodem123")],
                "tty.wch*": [],
                "tty.SLAB*": [],
                "cu.usb*": [],
                "cu.wch*": [],
                "cu.SLAB*": [],
            }
            return matches[pattern]

    monkeypatch.setattr(lerobot_find_port, "Path", lambda _: FakeDevDir())

    assert lerobot_find_port.find_available_ports() == [
        "/dev/tty.usbmodem123",
        "/dev/ttyACM0",
        "/dev/ttyUSB0",
    ]


def test_find_removed_port_returns_single_missing_port():
    assert (
        lerobot_find_port.find_removed_port(
            ["/dev/ttyACM0", "/dev/ttyACM1"],
            ["/dev/ttyACM1"],
        )
        == "/dev/ttyACM0"
    )


def test_find_removed_port_raises_when_no_port_disappeared():
    with pytest.raises(OSError, match="No device disappeared"):
        lerobot_find_port.find_removed_port(["/dev/ttyACM0"], ["/dev/ttyACM0"])


def test_find_removed_port_raises_when_multiple_ports_disappeared():
    with pytest.raises(OSError, match="More than one device disappeared"):
        lerobot_find_port.find_removed_port(
            ["/dev/ttyACM0", "/dev/ttyACM1"],
            [],
        )
