#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys
import time
from typing import List

import adafruit_vl53l1x
import board
import lgpio

XSHUTS = [7, 23, 21, 27]
ADDRS = [0x30, 0x31, 0x32, 0x33]
LABELS = ["zadni_pravy", "pravy_predni", "levy_zadni", "levy_predni"]

BOOT_DELAY = 2.0
XSHUT_DELAY = 1.0
PROBE_ATTEMPTS = 30
PROBE_DELAY = 0.1
SKIP_MISSING = True

MODE = "LONG"
MODE_MAP = {"SHORT": 1, "LONG": 2}
TIMING_MS = 100
EMA = 0.6
LOOP_DELAY = 0.2


def init_sensor(i2c: board.I2C, address: int) -> adafruit_vl53l1x.VL53L1X:
    posledni_chyba: Exception | None = None
    for _ in range(max(1, PROBE_ATTEMPTS)):
        try:
            return adafruit_vl53l1x.VL53L1X(i2c, address=address)
        except Exception as chyba:
            posledni_chyba = chyba
            time.sleep(PROBE_DELAY)
    assert posledni_chyba is not None
    raise posledni_chyba


def readdress(i2c: board.I2C) -> tuple[List[int], List[str]]:
    cip = lgpio.gpiochip_open(0)
    aktivni_adresy: List[int] = []
    aktivni_popisky: List[str] = []
    try:
        for pin in XSHUTS:
            lgpio.gpio_claim_output(cip, pin)
            lgpio.gpio_write(cip, pin, 0)
        time.sleep(0.05)

        for popisek, pin, adresa in zip(LABELS, XSHUTS, ADDRS):
            lgpio.gpio_write(cip, pin, 1)
            time.sleep(XSHUT_DELAY)
            try:
                sensor = init_sensor(i2c, 0x29)
                if adresa != 0x29:
                    sensor.set_address(adresa)
                    time.sleep(0.02)
                    sensor = init_sensor(i2c, adresa)
                print(f"{popisek}: BCM{pin} -> 0x{adresa:02X}")
                aktivni_adresy.append(adresa)
                aktivni_popisky.append(popisek)
            except Exception as chyba:
                print(f"{popisek}: BCM{pin} -> missing ({chyba})", file=sys.stderr)
                if not SKIP_MISSING:
                    raise
                lgpio.gpio_write(cip, pin, 0)
    finally:
        lgpio.gpiochip_close(cip)
    return aktivni_adresy, aktivni_popisky


def setup_sensors(i2c: board.I2C, addrs: List[int]) -> List[adafruit_vl53l1x.VL53L1X]:
    cidla: List[adafruit_vl53l1x.VL53L1X] = []
    for adresa in addrs:
        sensor = init_sensor(i2c, adresa)
        sensor.distance_mode = MODE_MAP[MODE]
        sensor.timing_budget = max(15, min(500, TIMING_MS))
        sensor.start_ranging()
        cidla.append(sensor)
    return cidla


def main() -> None:
    if BOOT_DELAY > 0:
        time.sleep(BOOT_DELAY)

    i2c = board.I2C()
    addrs, labels = readdress(i2c)

    if "--readdress-only" in sys.argv:
        return

    cidla = setup_sensors(i2c, addrs)

    def _cleanup(*_args) -> None:
        for cidlo in cidla:
            try:
                cidlo.stop_ranging()
            except Exception:
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    col_width = max(len(label) for label in labels) + 14
    header = " | ".join(
        f"{labels[idx]}@0x{addrs[idx]:02X}".ljust(col_width) for idx in range(len(labels))
    )
    print(header)
    print("-" * len(header))

    posledni_hodnoty: List[float | None] = [None for _ in cidla]
    filtrovane_hodnoty: List[float | None] = [None for _ in cidla]
    ema = max(0.0, min(1.0, EMA))

    while True:
        for idx, cidlo in enumerate(cidla):
            if not cidlo.data_ready:
                continue
            hodnota = cidlo.distance
            cidlo.clear_interrupt()
            posledni_hodnoty[idx] = hodnota
            if ema <= 0.0:
                filtrovane_hodnoty[idx] = hodnota
            else:
                minula = filtrovane_hodnoty[idx]
                filtrovane_hodnoty[idx] = hodnota if minula is None else (ema * hodnota + (1 - ema) * minula)

        sloupce: List[str] = []
        for idx in range(len(cidla)):
            hodnota = filtrovane_hodnoty[idx] if ema > 0.0 else posledni_hodnoty[idx]
            text = "---" if hodnota is None else f"{hodnota * 10:.1f} mm"
            sloupce.append(f"{text:<{col_width}}")
        print(" | ".join(sloupce))
        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    main()
