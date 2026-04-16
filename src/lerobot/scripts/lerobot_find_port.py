# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
lerobot-find-port
```
"""

import os
import time
from pathlib import Path

from serial.tools import list_ports

USB_SERIAL_GLOBS = (
    "ttyACM*",
    "ttyUSB*",
    "ttyAMA*",
    "tty.usb*",
    "tty.wch*",
    "tty.SLAB*",
    "cu.usb*",
    "cu.wch*",
    "cu.SLAB*",
)


def find_available_ports() -> list[str]:
    """List serial ports with pyserial first, then fall back to likely USB serial device names."""
    ports = sorted({port.device for port in list_ports.comports()})
    if ports:
        return ports

    if os.name == "nt":
        return []

    fallback_ports: set[str] = set()
    dev_dir = Path("/dev")
    for pattern in USB_SERIAL_GLOBS:
        fallback_ports.update(str(path) for path in dev_dir.glob(pattern))

    return sorted(fallback_ports)


def find_removed_port(ports_before: list[str], ports_after: list[str]) -> str:
    """Return the single disconnected port or raise a detailed error."""
    ports_diff = sorted(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        return ports_diff[0]
    if len(ports_diff) == 0:
        raise OSError(
            "Could not detect the port. No device disappeared after disconnecting the MotorsBus. "
            f"Ports before: {ports_before}. Ports after: {ports_after}."
        )

    raise OSError(
        "Could not detect the port. More than one device disappeared after disconnecting the MotorsBus. "
        f"Removed ports: {ports_diff}."
    )


def find_port():
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    print("Ports before disconnecting:", ports_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()  # Wait for user to disconnect the device

    time.sleep(0.5)  # Allow some time for port to be released
    ports_after = find_available_ports()
    port = find_removed_port(ports_before, ports_after)
    print(f"The port of this MotorsBus is '{port}'")
    print("Reconnect the USB cable.")


def main():
    find_port()


if __name__ == "__main__":
    main()
