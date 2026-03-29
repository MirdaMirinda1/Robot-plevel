#!/usr/bin/env python3

from __future__ import annotations

import argparse
import errno
import glob
import os
import signal
import sys
import threading
import time
from typing import Optional

import RPi.GPIO as GPIO
from dotenv import load_dotenv
from gpiozero import Button, Device, OutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

from app.weed_spray import (
    FRONT_CROP_BOTTOM_RATIO,
    FRONT_CROP_TOP_RATIO,
    FRONT_MIN_GREEN_RATIO,
    FRONT_MIN_PATH_RATIO,
    FRONT_ZONE_WIDTH_RATIO,
    SIDE_CROP_BOTTOM_RATIO,
    SIDE_CROP_TOP_RATIO,
    SIDE_MIN_GREEN_RATIO,
    SIDE_MIN_PATH_RATIO,
    SIDE_ZONE_WIDTH_RATIO,
    WeedDetector,
    open_front_usb,
    open_side_cameras,
)
from app.motory import LEFT_DEFAULT, RIGHT_DEFAULT, L298NDriver, pins_from_env

ACTIVE_HIGH = False
RELE = {
    "levy": 4,
    "pravy": 9,
    "predni": 14,
    "cerpadlo": 16,
}

PRIKAZY: dict[str, tuple[str, bool]] = {
    "PRA": ("pravy", True),
    "pra": ("pravy", False),
    "LEV": ("levy", True),
    "lev": ("levy", False),
    "PRE": ("predni", True),
    "pre": ("predni", False),
    "CER": ("cerpadlo", True),
    "cer": ("cerpadlo", False),
}
AUTO_ON = "AUT"
AUTO_OFF = "aut"
RECORD_ON = "REC"
RECORD_OFF = "rec"


def _udelej_zarizeni() -> dict[str, OutputDevice]:
    Device.pin_factory = LGPIOFactory()
    return {
        nazev: OutputDevice(pin, active_high=ACTIVE_HIGH, initial_value=False)
        for nazev, pin in RELE.items()
    }


def _proved_prikaz(prikaz: str, zarizeni: dict[str, OutputDevice]) -> None:
    cil = PRIKAZY.get(prikaz)
    if not cil:
        return
    nazev, zapnout = cil
    rele = zarizeni[nazev]
    if zapnout:
        rele.on()
        print(f"{prikaz} -> {nazev} ON")
    else:
        rele.off()
        print(f"{prikaz} -> {nazev} OFF")


def _precti_otacky_ventilatoru() -> int:
    cesty = glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input")
    for cesta in cesty:
        try:
            with open(cesta, "r", encoding="ascii") as soubor:
                return int(soubor.read().strip() or "0")
        except (OSError, ValueError):
            continue
    return 0


def _rfcomm_send(
    path: str,
    payload: bytes,
    connected: threading.Event,
    write_lock: threading.Lock,
) -> None:
    deskriptor = None
    try:
        with write_lock:
            deskriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            os.write(deskriptor, payload)
    except OSError as exc:
        if exc.errno not in {errno.EAGAIN, errno.EWOULDBLOCK}:
            connected.clear()
    finally:
        if deskriptor is not None:
            try:
                os.close(deskriptor)
            except OSError:
                pass


def _fan_sender(
    path: str,
    connected: threading.Event,
    stop: threading.Event,
    write_lock: threading.Lock,
) -> None:
    while not stop.is_set():
        if not connected.is_set():
            time.sleep(0.2)
            continue
        otacky = _precti_otacky_ventilatoru()
        payload = f"*F{otacky}\n".encode("ascii")
        _rfcomm_send(path, payload, connected, write_lock)
        time.sleep(1.0)


class FrontObstacleGuard:
    def __init__(
        self,
        threshold_cm: float = 20.0,
        trig_pin: int = 8,
        echo_pins: tuple[int, int, int] = (22, 18, 25),
        interval_s: float = 0.08,
        max_pulse_time: float = 0.04,
    ):
        self.threshold_cm = float(threshold_cm)
        self.trig_pin = trig_pin
        self.echo_pins = echo_pins
        self.interval_s = interval_s
        self.max_pulse_time = max_pulse_time
        self._speed_of_sound_cm_s = 34300.0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._obstacle = False
        self._distances: tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)
        self._ready = False
        self._gpio_mode = "rpi"
        self._gpio_handle = None

    @property
    def obstacle_ahead(self) -> bool:
        with self._lock:
            return self._obstacle

    @property
    def distances_cm(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        with self._lock:
            return self._distances

    def start(self) -> None:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            for pin in self.echo_pins:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.output(self.trig_pin, GPIO.LOW)
            self._gpio_mode = "rpi"
        except Exception:
            import lgpio

            self._gpio_handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._gpio_handle, self.trig_pin, 0)
            for pin in self.echo_pins:
                lgpio.gpio_claim_input(self._gpio_handle, pin)
            self._gpio_mode = "lgpio"
        time.sleep(0.05)
        self._ready = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _write_trig(self, value: int) -> None:
        if self._gpio_mode == "lgpio":
            import lgpio

            assert self._gpio_handle is not None
            lgpio.gpio_write(self._gpio_handle, self.trig_pin, value)
            return
        GPIO.output(self.trig_pin, GPIO.HIGH if value else GPIO.LOW)

    def _read_echo(self, pin: int) -> int:
        if self._gpio_mode == "lgpio":
            import lgpio

            assert self._gpio_handle is not None
            return int(lgpio.gpio_read(self._gpio_handle, pin))
        return int(GPIO.input(pin))

    def _measure_once(self) -> list[Optional[float]]:
        self._write_trig(1)
        time.sleep(0.00001)
        self._write_trig(0)

        last_state = {pin: self._read_echo(pin) for pin in self.echo_pins}
        start_times: dict[int, Optional[float]] = {pin: None for pin in self.echo_pins}
        end_times: dict[int, Optional[float]] = {pin: None for pin in self.echo_pins}

        deadline = time.monotonic() + self.max_pulse_time
        while time.monotonic() < deadline:
            now = time.monotonic()
            for pin in self.echo_pins:
                state = self._read_echo(pin)
                if last_state[pin] == 0 and state == 1:
                    start_times[pin] = now
                elif last_state[pin] == 1 and state == 0 and start_times[pin] is not None:
                    end_times[pin] = now
                last_state[pin] = state
            if all(end_times[pin] is not None for pin in self.echo_pins):
                break

        distances: list[Optional[float]] = []
        for pin in self.echo_pins:
            start = start_times[pin]
            end = end_times[pin]
            if start is None or end is None:
                distances.append(None)
            else:
                pulse_duration = end - start
                distances.append((pulse_duration * self._speed_of_sound_cm_s) / 2.0)
        return distances

    def _loop(self) -> None:
        while not self._stop.is_set():
            distances = self._measure_once()
            obstacle = any(d is not None and d <= self.threshold_cm for d in distances)
            with self._lock:
                self._obstacle = obstacle
                self._distances = (distances[0], distances[1], distances[2])
            time.sleep(self.interval_s)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._ready:
            if self._gpio_mode == "lgpio":
                try:
                    import lgpio

                    if self._gpio_handle is not None:
                        lgpio.gpiochip_close(self._gpio_handle)
                except Exception:
                    pass
                self._gpio_handle = None
            else:
                GPIO.cleanup((self.trig_pin, *self.echo_pins))
            self._ready = False


class WaterLevelGuard:
    def __init__(self, pin: int = 24) -> None:
        self.sensor = Button(pin)

    @property
    def has_water(self) -> bool:
        return not bool(self.sensor.is_pressed)

    def close(self) -> None:
        try:
            self.sensor.close()
        except Exception:
            pass


class SprayController:
    def __init__(
        self,
        devices: dict[str, OutputDevice],
        water_guard: WaterLevelGuard,
        spray_seconds: float = 1.50,
        settle_seconds: float = 0.12,
        loop_delay: float = 0.08,
        request_cooldown_s: float = 3.00,
        after_spray_cooldown_s: float = 5.00,
    ) -> None:
        self.devices = devices
        self.water_guard = water_guard
        self.spray_seconds = spray_seconds
        self.settle_seconds = settle_seconds
        self.loop_delay = loop_delay
        self.request_cooldown_s = request_cooldown_s
        self.after_spray_cooldown_s = after_spray_cooldown_s
        self.cooldown_map = {
            "levy": float(os.environ.get("WEED_COOLDOWN_LEFT", "3.0")),
            "pravy": float(os.environ.get("WEED_COOLDOWN_RIGHT", "3.0")),
            "predni": float(os.environ.get("WEED_COOLDOWN_FRONT", "6.0")),
        }
        self.spray_map = {
            "levy": float(os.environ.get("WEED_SPRAY_LEFT", "1.5")),
            "pravy": float(os.environ.get("WEED_SPRAY_RIGHT", "1.5")),
            "predni": float(os.environ.get("WEED_SPRAY_FRONT", "2.0")),
        }
        self.front_detector = WeedDetector(
            min_path_ratio=FRONT_MIN_PATH_RATIO,
            min_green_ratio=FRONT_MIN_GREEN_RATIO,
            crop_top_ratio=FRONT_CROP_TOP_RATIO,
            crop_bottom_ratio=FRONT_CROP_BOTTOM_RATIO,
            zone_width_ratio=FRONT_ZONE_WIDTH_RATIO,
        )
        self.left_detector = WeedDetector(
            min_path_ratio=SIDE_MIN_PATH_RATIO,
            min_green_ratio=SIDE_MIN_GREEN_RATIO,
            crop_top_ratio=SIDE_CROP_TOP_RATIO,
            crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
            zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
        )
        self.right_detector = WeedDetector(
            min_path_ratio=SIDE_MIN_PATH_RATIO,
            min_green_ratio=SIDE_MIN_GREEN_RATIO,
            crop_top_ratio=SIDE_CROP_TOP_RATIO,
            crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
            zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
        )
        self.front_cam = None
        self.left_cam = None
        self.right_cam = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._water_empty_logged = False
        self._lock = threading.Lock()
        self._pending_valve: Optional[str] = None
        self._last_spray = {"levy": 0.0, "predni": 0.0, "pravy": 0.0}

    def start(self) -> None:
        self.front_cam = open_front_usb()
        self.left_cam, self.right_cam = open_side_cameras()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("AUT -> postrik bezi.")

    def _set_outputs(self, active_valve: Optional[str]) -> None:
        left_on = active_valve == "levy"
        front_on = active_valve == "predni"
        right_on = active_valve == "pravy"
        self.devices["levy"].on() if left_on else self.devices["levy"].off()
        self.devices["predni"].on() if front_on else self.devices["predni"].off()
        self.devices["pravy"].on() if right_on else self.devices["pravy"].off()
        self.devices["cerpadlo"].on() if active_valve is not None else self.devices["cerpadlo"].off()

    def consume_request(self) -> Optional[str]:
        with self._lock:
            valve = self._pending_valve
            self._pending_valve = None
            return valve

    def spray_burst(self, valve: str, sleep_fn) -> bool:
        if not self.water_guard.has_water:
            print("AUT -> voda dosla, postrik preskakuji.")
            self._set_outputs(None)
            return False
        duration = self.spray_map.get(valve, self.spray_seconds)
        print(f"AUT -> strikam {valve} {duration:.2f}s")
        self._set_outputs(valve)
        try:
            if not sleep_fn(duration):
                return False
        finally:
            self._set_outputs(None)
        with self._lock:
            self._pending_valve = None
            self._last_spray[valve] = time.monotonic()
        return sleep_fn(self.settle_seconds)

    def _loop(self) -> None:
        last_log = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if not self.water_guard.has_water:
                if not self._water_empty_logged:
                    print("AUT -> voda dosla, postrik vypnuty.")
                    self._water_empty_logged = True
                self._set_outputs(None)
                with self._lock:
                    self._pending_valve = None
                time.sleep(0.2)
                continue

            self._water_empty_logged = False
            front = self.front_cam.read() if self.front_cam is not None else None
            left = self.left_cam.read() if self.left_cam is not None else None
            right = self.right_cam.read() if self.right_cam is not None else None
            front_det = self.front_detector.detect(front)
            left_det = self.left_detector.detect(left)
            right_det = self.right_detector.detect(right)

            candidates = []
            if left_det.plevel:
                candidates.append(("levy", left_det.skore))
            if front_det.plevel:
                candidates.append(("predni", front_det.skore))
            if right_det.plevel:
                candidates.append(("pravy", right_det.skore))

            queued_valve = None
            if candidates:
                priority = {"predni": 2, "levy": 1, "pravy": 1}
                available = []
                for name, score in candidates:
                    last = self._last_spray.get(name, 0.0)
                    cooldown = self.cooldown_map.get(name, self.after_spray_cooldown_s)
                    if (now - last) >= cooldown:
                        available.append((name, score))
                if available:
                    queued_valve, _ = max(available, key=lambda item: (item[1], priority.get(item[0], 0)))
                    with self._lock:
                        if self._pending_valve is None:
                            self._pending_valve = queued_valve

            if now - last_log >= 0.5:
                with self._lock:
                    pending = self._pending_valve
                print(
                    "AUT spray -> "
                    f"L {left_det.skore:.4f}/{left_det.skore_chodniku:.2f} "
                    f"F {front_det.skore:.4f}/{front_det.skore_chodniku:.2f} "
                    f"P {right_det.skore:.4f}/{right_det.skore_chodniku:.2f} "
                    f"voda={int(self.water_guard.has_water)} pending={pending or '-'}"
                )
                last_log = now
            time.sleep(self.loop_delay)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._set_outputs(None)
        for cam in (self.front_cam, self.left_cam, self.right_cam):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
        self.front_cam = None
        self.left_cam = None
        self.right_cam = None


class SideGuide:
    SENSOR_ADDRS = {
        "right_rear": 0x30,
        "right_front": 0x31,
        "left_rear": 0x32,
        "left_front": 0x33,
    }

    def __init__(
        self,
        ema: float = 0.55,
        loop_delay: float = 0.05,
        valid_min_cm: float = 3.0,
        valid_max_cm: float = 80.0,
        kp_dist: float = 0.014,
        kp_angle: float = 0.018,
        max_correction: float = 0.16,
        roi_size: int = 8,
    ):
        self.ema = max(0.0, min(1.0, ema))
        self.loop_delay = loop_delay
        self.valid_min_cm = valid_min_cm
        self.valid_max_cm = valid_max_cm
        self.kp_dist = kp_dist
        self.kp_angle = kp_angle
        self.max_correction = max_correction
        self.roi_size = max(4, min(16, roi_size))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._i2c = None
        self._sensors: dict[str, adafruit_vl53l1x.VL53L1X] = {}
        self._values: dict[str, Optional[float]] = {name: None for name in self.SENSOR_ADDRS}

    def start(self) -> None:
        try:
            import adafruit_vl53l1x
            import board
        except ImportError:
            extra_site = os.environ.get(
                "BT_RELAYS_EXTRA_SITE",
                "/home/pi/robot/.venv/lib/python3.13/site-packages",
            )
            if extra_site and extra_site not in sys.path:
                sys.path.append(extra_site)
            import adafruit_vl53l1x
            import board

        self._i2c = board.I2C()
        last_exc = None
        for _attempt in range(20):
            try:
                for name, addr in self.SENSOR_ADDRS.items():
                    sensor = adafruit_vl53l1x.VL53L1X(self._i2c, address=addr)
                    sensor.distance_mode = 2
                    sensor.timing_budget = 50
                    sensor.roi_xy = (self.roi_size, self.roi_size)
                    sensor.roi_center = 199
                    sensor.start_ranging()
                    self._sensors[name] = sensor
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                for sensor in self._sensors.values():
                    try:
                        sensor.stop_ranging()
                    except Exception:
                        pass
                self._sensors.clear()
                time.sleep(0.25)
        if last_exc is not None:
            raise last_exc
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            updated: dict[str, Optional[float]] = {}
            for name, sensor in self._sensors.items():
                value = None
                try:
                    if sensor.data_ready:
                        value = sensor.distance
                        sensor.clear_interrupt()
                except Exception:
                    value = None
                if value is not None:
                    prev = self._values[name]
                    if prev is None:
                        updated[name] = value
                    else:
                        updated[name] = self.ema * value + (1.0 - self.ema) * prev
            if updated:
                with self._lock:
                    self._values.update(updated)
            time.sleep(self.loop_delay)

    def snapshot(self) -> dict[str, Optional[float]]:
        with self._lock:
            return dict(self._values)

    def _valid_pair(self, side: str) -> tuple[Optional[float], Optional[float]]:
        values = self.snapshot()
        front = values[f"{side}_front"]
        rear = values[f"{side}_rear"]
        if front is not None and not (self.valid_min_cm <= front <= self.valid_max_cm):
            front = None
        if rear is not None and not (self.valid_min_cm <= rear <= self.valid_max_cm):
            rear = None
        return front, rear

    def choose_side(self) -> tuple[Optional[str], Optional[float]]:
        left_front, left_rear = self._valid_pair("left")
        right_front, right_rear = self._valid_pair("right")

        left_values = [v for v in (left_front, left_rear) if v is not None]
        right_values = [v for v in (right_front, right_rear) if v is not None]

        if not left_values and not right_values:
            return None, None
        if len(left_values) > len(right_values):
            return "left", sum(left_values) / len(left_values)
        if len(right_values) > len(left_values):
            return "right", sum(right_values) / len(right_values)

        left_avg = sum(left_values) / len(left_values) if left_values else 999.0
        right_avg = sum(right_values) / len(right_values) if right_values else 999.0
        if left_avg <= right_avg:
            return "left", left_avg
        return "right", right_avg

    def correction_for(self, side: str, target_cm: float) -> float:
        front, rear = self._valid_pair(side)
        values = [v for v in (front, rear) if v is not None]
        if not values:
            return 0.0
        avg = sum(values) / len(values)
        dist_term = self.kp_dist * (avg - target_cm)
        angle_term = 0.0
        if front is not None and rear is not None:
            angle_term = self.kp_angle * (front - rear)
        correction = dist_term + angle_term
        if side == "right":
            correction = -correction
        return max(-self.max_correction, min(self.max_correction, correction))

    def alignment_pair(self, side: str) -> tuple[Optional[float], Optional[float]]:
        return self._valid_pair(side)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        for sensor in self._sensors.values():
            try:
                sensor.stop_ranging()
            except Exception:
                pass




def _speed_from_slider(value: int) -> float:
    v = max(0, min(100, value))
    return (v - 50) / 50.0


def _sensor_sender(
    path: str,
    guard: FrontObstacleGuard,
    connected: threading.Event,
    stop: threading.Event,
    write_lock: threading.Lock,
) -> None:
    while not stop.is_set():
        if not connected.is_set():
            time.sleep(0.2)
            continue
        right, middle, left = guard.distances_cm
        right_val = 0 if right is None else int(round(right))
        middle_val = 0 if middle is None else int(round(middle))
        left_val = 0 if left is None else int(round(left))
        payload = f"*1{right_val}\n*2{middle_val}\n*3{left_val}\n".encode("ascii")
        _rfcomm_send(path, payload, connected, write_lock)
        time.sleep(0.2)


def _read_loop(
    path: str,
    devices: dict[str, OutputDevice],
    driver: L298NDriver,
    guard: FrontObstacleGuard,
    side_guide: Optional[SideGuide],
    water_guard: WaterLevelGuard,
    connected: threading.Event,
) -> int:
    relay_buffer = ""
    slider_active = False
    slider_buf = ""
    pending_motor: Optional[str] = None
    pending_motor_time = 0.0
    pending_slider = False
    pending_slider_time = 0.0

    speed = 0.0
    left_pressed = False
    right_pressed = False
    last_left = 0.0
    last_right = 0.0
    blocked_logged = False
    recording = False
    route: list[tuple[float, float, float]] = []
    segment_started_at = time.monotonic()
    replay_thread: Optional[threading.Thread] = None
    replay_stop = threading.Event()
    replay_forward_active = threading.Event()
    manual_watch: Optional[threading.Thread] = None
    spray: Optional[SprayController] = None
    turn_speed_left = 0.48
    turn_speed_right = -0.48
    turn_duration_s = 3.25
    manual_adjust_delay_s = 2.5
    manual_release_quiet_s = 0.45
    vl53_search_turn_speed = 0.42
    vl53_search_step_s = 0.22
    vl53_search_settle_s = 0.06
    manual_adjust_turn_speed = 0.48
    pomocny_pomer = 0.25
    prekazka_uvolni_s = 0.4
    manual_slew_per_s = 3.0
    aut_trim_sila = 0.28
    manual_adjust_active = threading.Event()
    manual_adjust_seen_input = threading.Event()
    manual_adjust_last_input = 0.0
    blokuj_dopredu = False
    posledni_prekazka = 0.0
    smooth_left = 0.0
    smooth_right = 0.0
    smooth_last_ts = time.monotonic()
    trim_lock = threading.Lock()

    def _reset_manual_state() -> None:
        nonlocal speed, left_pressed, right_pressed, last_left, last_right, blocked_logged, segment_started_at
        speed = 0.0
        left_pressed = False
        right_pressed = False
        last_left = 0.0
        last_right = 0.0
        blocked_logged = False
        segment_started_at = time.monotonic()

    def _replay_running() -> bool:
        nonlocal replay_thread
        if replay_thread is None:
            return False
        if replay_thread.is_alive():
            return True
        replay_thread = None
        return False

    def _record_current_segment(now: float) -> None:
        nonlocal segment_started_at
        if not recording:
            segment_started_at = now
            return
        dt = now - segment_started_at
        if dt > 0.0:
            route.append((dt, last_left, last_right))
        segment_started_at = now

    def _stop_recording() -> None:
        nonlocal recording
        if not recording:
            return
        _record_current_segment(time.monotonic())
        recording = False
        print(f"{RECORD_OFF} -> recording OFF ({len(route)} segmentu)")

    def _start_recording() -> None:
        nonlocal recording, route, segment_started_at
        if _replay_running():
            _stop_replay()
        route = []
        recording = True
        segment_started_at = time.monotonic()
        print(f"{RECORD_ON} -> recording ON")

    def _sleep_interruptible(duration: float) -> bool:
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            if replay_forward_active.is_set() and guard.obstacle_ahead:
                driver.zastav()
                while guard.obstacle_ahead and not replay_stop.is_set() and connected.is_set():
                    time.sleep(0.02)
            if replay_stop.is_set() or not connected.is_set():
                return False
            zbyva = deadline - time.monotonic()
            if zbyva <= 0:
                break
            time.sleep(min(0.02, zbyva))
        return True

    def _note_manual_adjust_input() -> None:
        nonlocal manual_adjust_last_input
        if not manual_adjust_active.is_set():
            return
        manual_adjust_last_input = time.monotonic()
        manual_adjust_seen_input.set()

    def _manual_adjust_done() -> bool:
        if not manual_adjust_seen_input.is_set():
            return False
        if left_pressed or right_pressed:
            return False
        if abs(last_left) > 1e-3 or abs(last_right) > 1e-3:
            return False
        return (time.monotonic() - manual_adjust_last_input) >= manual_release_quiet_s

    def _run_turn(left: float, right: float, duration_s: float) -> bool:
        driver.nastav(left, right)
        return _sleep_interruptible(duration_s)

    def _search_vl53_alignment() -> bool:
        if side_guide is None:
            return False

        offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        current_offset = 0

        def _move_to_offset(target_offset: int) -> bool:
            nonlocal current_offset
            delta = target_offset - current_offset
            if delta == 0:
                return True
            turn = vl53_search_turn_speed if delta > 0 else -vl53_search_turn_speed
            if not _run_turn(turn, -turn, abs(delta) * vl53_search_step_s):
                return False
            driver.zastav()
            current_offset = target_offset
            return _sleep_interruptible(vl53_search_settle_s)

        try:
            for target_offset in offsets:
                if not _move_to_offset(target_offset):
                    return False
                side, target = side_guide.choose_side()
                if side is None or target is None:
                    print(f"AUT -> VL53 search {target_offset:+d}: obrubnik nevidim")
                    continue
                front_dist, rear_dist = side_guide.alignment_pair(side)
                if front_dist is None or rear_dist is None:
                    print(f"AUT -> VL53 search {target_offset:+d}: {side} nema kompletni par")
                    continue
                diff = front_dist - rear_dist
                print(
                    "AUT -> VL53 search "
                    f"{target_offset:+d}: side={side} front={front_dist:.1f} rear={rear_dist:.1f} "
                    f"diff={diff:.1f} target={target:.1f}"
                )
                if abs(diff) <= 1.0:
                    print(
                        "AUT -> VL53 srovnano "
                        f"({side}: front={front_dist:.1f} rear={rear_dist:.1f})"
                    )
                    return True
            print("AUT -> VL53 podobnou polohu nenasly, vracim se na zaklad po 180 otocce")
            return False
        finally:
            if current_offset != 0:
                _move_to_offset(0)
            driver.zastav()

    def _manual_adjust_phase(duration_s: Optional[float], wait_for_input: bool) -> bool:
        manual_adjust_seen_input.clear()
        driver.zastav()
        _reset_manual_state()
        manual_adjust_active.set()
        try:
            if duration_s is None:
                print("AUT -> cekam na manualni dorovnani. L = toc vlevo, P = toc vpravo, po pusteni pokracuji.")
            else:
                print(
                    f"AUT -> manualni dorovnani {duration_s:.1f} s. "
                    "L = toc vlevo, P = toc vpravo, po pusteni nebo po case pokracuji."
                )
            deadline = None if duration_s is None else time.monotonic() + max(0.0, duration_s)
            while True:
                if replay_stop.is_set() or not connected.is_set():
                    return False
                if _manual_adjust_done():
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    return True
                if not wait_for_input and manual_adjust_seen_input.is_set() and _manual_adjust_done():
                    return True
                time.sleep(0.02)
        finally:
            manual_adjust_active.clear()
            driver.zastav()
            _reset_manual_state()

    def _run_replay(snapshot: list[tuple[float, float, float]]) -> None:
        nonlocal spray
        replay_cur_left = 0.0
        replay_cur_right = 0.0
        voda_cekam = False
        def _soft_stop(from_left: float, from_right: float, duration_s: float = 0.25) -> bool:
            steps = max(1, int(duration_s / 0.05))
            for idx in range(steps, 0, -1):
                frac = idx / steps
                driver.nastav(from_left * frac, from_right * frac)
                if not _sleep_interruptible(duration_s / steps):
                    return False
            driver.zastav()
            return True

        try:
            driver.zastav()
            if not _sleep_interruptible(0.2):
                return
            driver.nastav(turn_speed_left, turn_speed_right)
            if not _sleep_interruptible(turn_duration_s):
                return
            driver.zastav()
            if not _sleep_interruptible(0.2):
                return

            if side_guide is not None:
                _search_vl53_alignment()
                if replay_stop.is_set() or not connected.is_set():
                    return

            if not _manual_adjust_phase(None, wait_for_input=True):
                return

            try:
                spray = SprayController(devices, water_guard)
                spray.start()
            except Exception as exc:
                spray = None
                print(f"AUT -> postrik se nespustil: {exc}")

            for dt, left, right in reversed(snapshot):
                if replay_stop.is_set() or not connected.is_set():
                    break
                if not water_guard.has_water:
                    if not voda_cekam:
                        print("AUT -> voda dosla, zastavuji a cekam.")
                        voda_cekam = True
                    driver.zastav()
                    while not water_guard.has_water and not replay_stop.is_set() and connected.is_set():
                        time.sleep(0.1)
                    if replay_stop.is_set() or not connected.is_set():
                        break
                    print("AUT -> voda ok, pokracuji.")
                    voda_cekam = False
                if spray is not None:
                    valve = spray.consume_request()
                    if valve is not None:
                        if not _soft_stop(replay_cur_left, replay_cur_right):
                            break
                        if not spray.spray_burst(valve, _sleep_interruptible):
                            break
                replay_left = right
                replay_right = left
                if manual_adjust_active.is_set():
                    pass
                else:
                    with trim_lock:
                        lp = left_pressed
                        rp = right_pressed
                    if lp and not rp:
                        replay_left -= aut_trim_sila
                        replay_right += aut_trim_sila
                    elif rp and not lp:
                        replay_left += aut_trim_sila
                        replay_right -= aut_trim_sila
                if abs(replay_left) > 1e-3 and abs(replay_right) <= 1e-3:
                    replay_right = replay_left * pomocny_pomer
                elif abs(replay_right) > 1e-3 and abs(replay_left) <= 1e-3:
                    replay_left = replay_right * pomocny_pomer
                replay_forward = ((replay_left + replay_right) / 2.0) < -0.05
                if replay_forward:
                    replay_forward_active.set()
                    if guard.obstacle_ahead:
                        print(f"AUT -> prekazka <= {guard.threshold_cm:.0f} cm, cekam.")
                else:
                    replay_forward_active.clear()
                driver.nastav(replay_left, replay_right)
                replay_cur_left = replay_left
                replay_cur_right = replay_right
                if not _sleep_interruptible(dt):
                    break
                replay_forward_active.clear()
        finally:
            if spray is not None:
                spray.stop()
                spray = None
            driver.zastav()

    def _stop_replay() -> None:
        nonlocal replay_thread
        replay_stop.set()
        if replay_thread is not None and replay_thread.is_alive():
            replay_thread.join(timeout=2.0)
        replay_thread = None
        replay_stop.clear()
        _reset_manual_state()
        driver.zastav()
        print(f"{AUTO_OFF} -> replay OFF")

    def _start_replay() -> None:
        nonlocal replay_thread
        if _replay_running():
            return
        if recording:
            _stop_recording()
        if not route:
            print(f"{AUTO_ON} -> neni zadna nahrana trasa")
            return
        _reset_manual_state()
        replay_stop.clear()
        replay_thread = threading.Thread(target=_run_replay, args=(list(route),), daemon=True)
        replay_thread.start()
        print(f"{AUTO_ON} -> replay ON")

    def _apply_alpha_command(cmd: str) -> None:
        if cmd in PRIKAZY:
            _proved_prikaz(cmd, devices)
        elif cmd == RECORD_ON:
            _start_recording()
        elif cmd == RECORD_OFF:
            _stop_recording()
        elif cmd == AUTO_ON:
            _start_replay()
        elif cmd == AUTO_OFF:
            _stop_replay()

    def _apply_motor_state() -> None:
        nonlocal last_left, last_right, blocked_logged, blokuj_dopredu, posledni_prekazka
        nonlocal smooth_left, smooth_right, smooth_last_ts
        if _replay_running() and not manual_adjust_active.is_set():
            return
        now = time.monotonic()
        motion_speed = speed
        if manual_adjust_active.is_set():
            blocked_logged = False
            if left_pressed and not right_pressed:
                left = -manual_adjust_turn_speed
                right = manual_adjust_turn_speed
            elif right_pressed and not left_pressed:
                left = manual_adjust_turn_speed
                right = -manual_adjust_turn_speed
            else:
                left = 0.0
                right = 0.0
            if left != last_left or right != last_right:
                driver.nastav(left, right)
                last_left = left
                last_right = right
            return

        if guard.obstacle_ahead:
            blokuj_dopredu = True
            posledni_prekazka = now
        elif blokuj_dopredu and (now - posledni_prekazka) >= prekazka_uvolni_s:
            blokuj_dopredu = False
        forward_blocked = motion_speed < 0 and blokuj_dopredu
        if forward_blocked:
            motion_speed = 0.0
            if not blocked_logged:
                print(f"Prekazka <= {guard.threshold_cm:.0f} cm: dopredny pohyb blokovan.")
                blocked_logged = True
        elif blocked_logged and not blokuj_dopredu:
            print("Prekazka pryc: dopredny pohyb povolen.")
            blocked_logged = False

        left = motion_speed if left_pressed else 0.0
        right = motion_speed if right_pressed else 0.0
        if left_pressed and not right_pressed:
            right = motion_speed * pomocny_pomer
        elif right_pressed and not left_pressed:
            left = motion_speed * pomocny_pomer

        if forward_blocked:
            driver.zastav()
            smooth_left = 0.0
            smooth_right = 0.0
            smooth_last_ts = now
            last_left = 0.0
            last_right = 0.0
            return

        dt = max(0.0, now - smooth_last_ts)
        max_step = manual_slew_per_s * dt if dt > 0 else 1.0
        def _step(cur: float, target: float) -> float:
            if target > cur:
                return min(target, cur + max_step)
            if target < cur:
                return max(target, cur - max_step)
            return cur

        smooth_left = _step(smooth_left, left)
        smooth_right = _step(smooth_right, right)
        smooth_last_ts = now
        if smooth_left != last_left or smooth_right != last_right:
            if recording:
                _record_current_segment(time.monotonic())
            driver.nastav(smooth_left, smooth_right)
            last_left = smooth_left
            last_right = smooth_right

    def _commit_motor(cmd: str) -> None:
        nonlocal left_pressed, right_pressed
        if cmd in {"L", "l"}:
            left_pressed = cmd.isupper()
        elif cmd in {"P", "p"}:
            right_pressed = cmd.isupper()
        _note_manual_adjust_input()
        if _replay_running() and not manual_adjust_active.is_set():
            with trim_lock:
                pass
            return
        _apply_motor_state()

    def _flush_pending(now: float) -> None:
        nonlocal pending_motor, pending_slider, relay_buffer
        if pending_motor and (now - pending_motor_time) > 0.05:
            _commit_motor(pending_motor)
            pending_motor = None
        if pending_slider and (now - pending_slider_time) > 0.05:
            relay_buffer = (relay_buffer + "A")[-3:]
            if relay_buffer in PRIKAZY or relay_buffer in {AUTO_ON, AUTO_OFF, RECORD_ON, RECORD_OFF}:
                _apply_alpha_command(relay_buffer)
                relay_buffer = ""
            pending_slider = False

    def _manual_obstacle_watch() -> None:
        nonlocal last_left, last_right, blocked_logged, blokuj_dopredu, posledni_prekazka
        while connected.is_set():
            if manual_adjust_active.is_set():
                time.sleep(0.02)
                continue
            forward_now = ((last_left + last_right) / 2.0) < -0.05
            if forward_now and guard.obstacle_ahead:
                blokuj_dopredu = True
                posledni_prekazka = time.monotonic()
                driver.zastav()
                last_left = 0.0
                last_right = 0.0
                if not blocked_logged:
                    print(f"Prekazka <= {guard.threshold_cm:.0f} cm: dopredny pohyb blokovan.")
                    blocked_logged = True
            time.sleep(0.02)

    while True:
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except FileNotFoundError:
            connected.clear()
            time.sleep(0.2)
            continue
        except PermissionError as exc:
            connected.clear()
            print(f"Nemám práva otevřít {path}: {exc}", file=sys.stderr)
            time.sleep(0.2)
            continue

        try:
            connected.set()
            if manual_watch is None or not manual_watch.is_alive():
                manual_watch = threading.Thread(target=_manual_obstacle_watch, daemon=True)
                manual_watch.start()
            while True:
                now = time.monotonic()
                _flush_pending(now)
                _apply_motor_state()

                try:
                    chunk = os.read(fd, 64)
                except BlockingIOError:
                    time.sleep(0.05)
                    continue

                if not chunk:
                    connected.clear()
                    _stop_recording()
                    _stop_replay()
                    _reset_manual_state()
                    driver.zastav()
                    break

                for c in chunk.decode("ascii", errors="ignore"):
                    now = time.monotonic()
                    _flush_pending(now)

                    if slider_active:
                        if c == "A":
                            if slider_buf:
                                try:
                                    speed = _speed_from_slider(int(slider_buf))
                                    _note_manual_adjust_input()
                                    _apply_motor_state()
                                except ValueError:
                                    pass
                            slider_active = False
                            slider_buf = ""
                            continue
                        if c.isdigit():
                            if len(slider_buf) < 3:
                                slider_buf += c
                            continue
                        slider_active = False
                        slider_buf = ""
                        continue

                    if pending_slider:
                        if c.isdigit():
                            slider_active = True
                            slider_buf = c
                            pending_slider = False
                            continue
                        relay_buffer = (relay_buffer + "A")[-3:]
                        if relay_buffer in PRIKAZY or relay_buffer in {AUTO_ON, AUTO_OFF, RECORD_ON, RECORD_OFF}:
                            _apply_alpha_command(relay_buffer)
                            relay_buffer = ""
                        pending_slider = False

                    if pending_motor:
                        if c.isalpha():
                            if pending_motor in {"P", "p"} and c in {"R", "r", "E", "e"}:
                                relay_buffer = (relay_buffer + pending_motor + c)[-3:]
                                if relay_buffer in PRIKAZY or relay_buffer in {AUTO_ON, AUTO_OFF, RECORD_ON, RECORD_OFF}:
                                    _apply_alpha_command(relay_buffer)
                                    relay_buffer = ""
                                pending_motor = None
                                continue
                            if pending_motor in {"L", "l"} and c in {"E", "e"}:
                                relay_buffer = (relay_buffer + pending_motor + c)[-3:]
                                if relay_buffer in PRIKAZY or relay_buffer in {AUTO_ON, AUTO_OFF, RECORD_ON, RECORD_OFF}:
                                    _apply_alpha_command(relay_buffer)
                                    relay_buffer = ""
                                pending_motor = None
                                continue
                        _commit_motor(pending_motor)
                        pending_motor = None

                    if c == "A":
                        pending_slider = True
                        pending_slider_time = now
                        continue

                    if c in {"L", "l", "P", "p"}:
                        pending_motor = c
                        pending_motor_time = now
                        continue

                    if c.isalpha():
                        relay_buffer = (relay_buffer + c)[-3:]
                        if relay_buffer in PRIKAZY or relay_buffer in {AUTO_ON, AUTO_OFF, RECORD_ON, RECORD_OFF}:
                            _apply_alpha_command(relay_buffer)
                            relay_buffer = ""
                    else:
                        relay_buffer = ""
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        time.sleep(0.05)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bluetooth ovládání relé přes /dev/rfcomm0.")
    parser.add_argument("--device", default="/dev/rfcomm0", help="Cesta k RFCOMM zařízení.")
    parser.add_argument("--front-stop-cm", type=float, default=35.0, help="Prah prekazky pro blokaci jizdy dopredu.")
    args = parser.parse_args()

    load_dotenv()

    devices = _udelej_zarizeni()
    driver = L298NDriver(
        pins_from_env("LEVY", LEFT_DEFAULT),
        pins_from_env("PRAVY", RIGHT_DEFAULT),
    )
    guard = FrontObstacleGuard(threshold_cm=args.front_stop_cm)
    water_guard = WaterLevelGuard()
    side_guide: Optional[SideGuide] = None
    connected = threading.Event()
    stop = threading.Event()
    write_lock = threading.Lock()

    def _shutdown(*_args) -> None:
        stop.set()
        connected.clear()
        driver.zastav()
        driver.close()
        guard.close()
        water_guard.close()
        if side_guide is not None:
            side_guide.close()
        for dev in devices.values():
            dev.off()
        print("Vše vypnuto.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    sender = threading.Thread(
        target=_fan_sender,
        args=(args.device, connected, stop, write_lock),
        daemon=True,
    )
    sender.start()
    sensor_stream = threading.Thread(
        target=_sensor_sender,
        args=(args.device, guard, connected, stop, write_lock),
        daemon=True,
    )
    sensor_stream.start()

    try:
        guard.start()
    except Exception as exc:
        print(f"Ultrazvukova ochrana se nespustila: {exc}", file=sys.stderr)
        print("Pokracuji bez blokace dopredne jizdy.", file=sys.stderr)

    try:
        side_guide = SideGuide()
        side_guide.start()
    except Exception as exc:
        side_guide = None
        print(f"Bocni VL53L1X se nespustily: {exc}", file=sys.stderr)
        print("Pokracuji bez korekce podle obrubniku.", file=sys.stderr)

    print("Čekám na Bluetooth příkazy...")
    try:
        return _read_loop(args.device, devices, driver, guard, side_guide, water_guard, connected)
    finally:
        driver.close()
        guard.close()
        water_guard.close()
        if side_guide is not None:
            side_guide.close()


if __name__ == "__main__":
    raise SystemExit(main())
