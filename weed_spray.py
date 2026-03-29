#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from gpiozero import Device, OutputDevice
from gpiozero.pins.lgpio import LGPIOFactory


ACTIVE_HIGH = False
RELE = {
    "levy": 4,
    "pravy": 9,
    "predni": 14,
    "cerpadlo": 16,
}

USB_CANDIDATE_SIZES = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 360),
]
CSI_SIZE = (640, 360)
USB_REQUIRE_WIDE = os.environ.get("USB_REQUIRE_WIDE", "1") == "1"
USB_MIN_ASPECT = float(os.environ.get("USB_MIN_ASPECT", "1.6"))
SIDE_ZONE_WIDTH_RATIO = float(os.environ.get("WEED_SIDE_ZONE_WIDTH_RATIO", "0.28"))
FRONT_ZONE_WIDTH_RATIO = float(os.environ.get("WEED_FRONT_ZONE_WIDTH_RATIO", "0.30"))
SIDE_CROP_TOP_RATIO = float(os.environ.get("WEED_SIDE_CROP_TOP_RATIO", "0.00"))
FRONT_CROP_TOP_RATIO = float(os.environ.get("WEED_FRONT_CROP_TOP_RATIO", "0.12"))
SIDE_CROP_BOTTOM_RATIO = float(os.environ.get("WEED_SIDE_CROP_BOTTOM_RATIO", "0.25"))
FRONT_CROP_BOTTOM_RATIO = float(os.environ.get("WEED_FRONT_CROP_BOTTOM_RATIO", "0.16"))
SIDE_MIN_PATH_RATIO = float(os.environ.get("WEED_SIDE_MIN_PATH_RATIO", "0.06"))
FRONT_MIN_PATH_RATIO = float(os.environ.get("WEED_FRONT_MIN_PATH_RATIO", "0.06"))
SIDE_MIN_GREEN_RATIO = float(os.environ.get("WEED_SIDE_MIN_GREEN_RATIO", "0.0002"))
FRONT_MIN_GREEN_RATIO = float(os.environ.get("WEED_FRONT_MIN_GREEN_RATIO", "0.0004"))


@dataclass
class Detection:
    plevel: bool
    skore: float
    skore_chodniku: float


@dataclass
class AnalysisResult:
    detekce: Detection
    obrazek: Optional[np.ndarray]


class RelayController:
    def __init__(self) -> None:
        Device.pin_factory = LGPIOFactory()
        self.zarizeni = {
            nazev: OutputDevice(pin, active_high=ACTIVE_HIGH, initial_value=False)
            for nazev, pin in RELE.items()
        }

    def set_valves(self, left: bool, front: bool, right: bool) -> None:
        self.zarizeni["levy"].on() if left else self.zarizeni["levy"].off()
        self.zarizeni["predni"].on() if front else self.zarizeni["predni"].off()
        self.zarizeni["pravy"].on() if right else self.zarizeni["pravy"].off()
        zapnout_cerpadlo = left or front or right
        self.zarizeni["cerpadlo"].on() if zapnout_cerpadlo else self.zarizeni["cerpadlo"].off()

    def close(self) -> None:
        for rele in self.zarizeni.values():
            try:
                rele.off()
            except Exception:
                pass
            try:
                rele.close()
            except Exception:
                pass


class UvcCamera:
    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = self._open_capture(path)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        vybrana_velikost = None
        for sirka_chci, vyska_chci in USB_CANDIDATE_SIZES:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, sirka_chci)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vyska_chci)
            time.sleep(0.15)
            ok, obrazek = self.cap.read()
            if ok and obrazek is not None and obrazek.size > 0:
                sirka = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vyska = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if vyska <= 0:
                    continue
                pomer_stran = sirka / vyska
                if USB_REQUIRE_WIDE and pomer_stran < USB_MIN_ASPECT:
                    continue
                vybrana_velikost = (int(sirka), int(vyska))
                break
        if not self.cap.isOpened():
            raise RuntimeError(f"USB kamera {path} nejde otevrit")
        ok, obrazek = self.cap.read()
        if not ok or obrazek is None:
            raise RuntimeError(f"USB kamera {path} nedava obraz")
        sirka = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vyska = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if USB_REQUIRE_WIDE and vyska > 0:
            pomer_stran = sirka / vyska
            if pomer_stran < USB_MIN_ASPECT:
                self.cap.release()
                raise RuntimeError(f"USB kamera {path} neni wide ({pomer_stran:.2f})")
        if vybrana_velikost is None:
            print(f"Predni USB kamera {path}: wide rezim se nevynutil, ponechavam {int(sirka)}x{int(vyska)}")
        else:
            print(f"Predni USB kamera {path}: {vybrana_velikost[0]}x{vybrana_velikost[1]}")

    @staticmethod
    def _open_capture(path: str):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()

        nazev = os.path.basename(path)
        if nazev.startswith("video") and nazev[5:].isdigit():
            index = int(nazev[5:])
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
        cap.release()
        raise RuntimeError(f"USB kamera {path} nejde otevrit")

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()


class PicamCamera:
    def __init__(self, camera_num: int, rotate_180: bool = True) -> None:
        from libcamera import ColorSpace
        from picamera2 import Picamera2

        self.rotate_180 = rotate_180
        self.picam = Picamera2(camera_num=camera_num)
        velikost_cipu = self.picam.camera_properties.get("PixelArraySize", CSI_SIZE)
        config = self.picam.create_video_configuration(
            main={"size": CSI_SIZE, "format": "BGR888"},
            controls={"ScalerCrop": (0, 0, int(velikost_cipu[0]), int(velikost_cipu[1]))},
            colour_space=ColorSpace.Srgb(),
        )
        self.picam.configure(config)
        self.picam.start()
        time.sleep(1.0)

    def read(self):
        frame = self.picam.capture_array("main")
        if frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def close(self) -> None:
        self.picam.close()


def _usb_candidates() -> list[str]:
    preferovane: list[str] = []
    ostatni: list[str] = []
    for cesta_k_adresari in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        adresar = Path(cesta_k_adresari)
        if not adresar.exists():
            continue
        for odkaz in sorted(adresar.iterdir()):
            nazev = odkaz.name.lower()
            if "usb" not in nazev:
                continue
            try:
                vyreseno = str(odkaz.resolve())
            except Exception:
                continue
            if "video-index0" in nazev:
                preferovane.append(vyreseno)
            else:
                ostatni.append(vyreseno)
    videne = set()
    spojene: list[str] = []
    for polozka in preferovane + ostatni:
        if polozka in videne:
            continue
        videne.add(polozka)
        spojene.append(polozka)

    for cesta in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            name = (cesta / "name").read_text(encoding="ascii").strip().lower()
        except Exception:
            continue
        if "usb" not in name and "uvc" not in name and "arducam" not in name:
            continue
        dev = f"/dev/{cesta.name}"
        if dev not in videne and os.path.exists(dev):
            videne.add(dev)
            spojene.append(dev)

    if spojene:
        return spojene

    return [f"/dev/video{i}" for i in range(16) if os.path.exists(f"/dev/video{i}")]


def open_front_usb():
    cesta_z_env = os.environ.get("WEED_FRONT_USB")
    kandidati = [cesta_z_env] if cesta_z_env else _usb_candidates()
    for cesta in kandidati:
        try:
            kamera = UvcCamera(cesta)
            print(f"Predni USB kamera: {cesta}")
            return kamera
        except Exception:
            continue
    raise RuntimeError("Predni USB kamera nenalezena")


def open_side_cameras():
    from picamera2 import Picamera2

    info_kamer = Picamera2.global_camera_info()
    indexy_bocnich = []
    for index, info in enumerate(info_kamer):
        text = " ".join(str(info.get(k, "")) for k in ("Model", "Id", "Location", "Pipeline", "Name")).lower()
        if "usb" in text or "uvc" in text:
            continue
        indexy_bocnich.append(index)
    if len(indexy_bocnich) < 2:
        raise RuntimeError("Nenasly se dve CSI kamery")

    levy_index = int(os.environ.get("WEED_LEFT_CSI_INDEX", indexy_bocnich[0]))
    pravy_index = int(os.environ.get("WEED_RIGHT_CSI_INDEX", indexy_bocnich[1]))
    print(f"Leva CSI kamera: {levy_index}")
    print(f"Prava CSI kamera: {pravy_index}")
    return PicamCamera(levy_index), PicamCamera(pravy_index)


class WeedDetector:
    def __init__(
        self,
        min_path_ratio: float = 0.08,
        min_green_ratio: float = 0.003,
        crop_top_ratio: float = 0.35,
        crop_bottom_ratio: float = 0.0,
        zone_width_ratio: float = 1.0,
    ):
        self.min_path_ratio = min_path_ratio
        self.min_green_ratio = min_green_ratio
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = max(0.0, min(0.9, crop_bottom_ratio))
        self.zone_width_ratio = max(0.05, min(1.0, zone_width_ratio))
        self.gray_lo = np.array([0, 0, 40], dtype=np.uint8)
        self.gray_hi = np.array([180, 75, 220], dtype=np.uint8)
        self.green_lo = np.array([35, 40, 40], dtype=np.uint8)
        self.green_hi = np.array([90, 255, 255], dtype=np.uint8)
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    def analyze(self, frame) -> AnalysisResult:
        if frame is None or frame.size == 0:
            return AnalysisResult(Detection(False, 0.0, 0.0), None)

        vyska, sirka = frame.shape[:2]
        y0 = int(vyska * self.crop_top_ratio)
        y1 = max(y0 + 1, int(vyska * (1.0 - self.crop_bottom_ratio)))
        sirka_zony = max(1, int(sirka * self.zone_width_ratio))
        x0 = max(0, (sirka - sirka_zony) // 2)
        x1 = min(sirka, x0 + sirka_zony)
        oblast = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(oblast, cv2.COLOR_BGR2HSV)
        obrazek = frame.copy()
        cv2.rectangle(obrazek, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)

        seda_maska = cv2.inRange(hsv, self.gray_lo, self.gray_hi)
        seda_maska = cv2.morphologyEx(seda_maska, cv2.MORPH_OPEN, self.kernel)
        seda_maska = cv2.morphologyEx(seda_maska, cv2.MORPH_CLOSE, self.kernel)

        kontury, _ = cv2.findContours(seda_maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not kontury:
            return AnalysisResult(Detection(False, 0.0, 0.0), obrazek)

        nejvetsi_kontura = max(kontury, key=cv2.contourArea)
        maska_chodniku = np.zeros(seda_maska.shape, dtype=np.uint8)
        cv2.drawContours(maska_chodniku, [nejvetsi_kontura], -1, 255, thickness=-1)
        plocha_chodniku = float(cv2.countNonZero(maska_chodniku))
        plocha_zony = float(maska_chodniku.shape[0] * maska_chodniku.shape[1])
        pomer_chodniku = plocha_chodniku / max(1.0, plocha_zony)
        if pomer_chodniku < self.min_path_ratio:
            kontura_v_obraze = nejvetsi_kontura.copy()
            kontura_v_obraze[:, 0, 0] += x0
            kontura_v_obraze[:, 0, 1] += y0
            cv2.drawContours(obrazek, [kontura_v_obraze], -1, (255, 0, 0), thickness=2)
            return AnalysisResult(Detection(False, 0.0, pomer_chodniku), obrazek)

        zelena_maska = cv2.inRange(hsv, self.green_lo, self.green_hi)
        zelena_maska = cv2.morphologyEx(zelena_maska, cv2.MORPH_OPEN, self.kernel)
        zelena_na_chodniku = cv2.bitwise_and(zelena_maska, maska_chodniku)
        zelene_pixely = float(cv2.countNonZero(zelena_na_chodniku))
        pomer_zelene = zelene_pixely / max(1.0, plocha_chodniku)
        detekce = Detection(pomer_zelene >= self.min_green_ratio, pomer_zelene, pomer_chodniku)

        kontura_v_obraze = nejvetsi_kontura.copy()
        kontura_v_obraze[:, 0, 0] += x0
        kontura_v_obraze[:, 0, 1] += y0
        cv2.drawContours(obrazek, [kontura_v_obraze], -1, (255, 0, 0), thickness=2)
        zona_v_obraze = obrazek[y0:y1, x0:x1]
        zona_v_obraze[zelena_na_chodniku > 0] = (
            0.35 * zona_v_obraze[zelena_na_chodniku > 0] + 0.65 * np.array([0, 255, 0])
        ).astype(np.uint8)
        popisek = f"weed={int(detekce.plevel)} green={detekce.skore:.4f} path={detekce.skore_chodniku:.2f}"
        cv2.putText(obrazek, popisek, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return AnalysisResult(detekce, obrazek)

    def detect(self, frame) -> Detection:
        return self.analyze(frame).detekce


def main() -> int:
    parser = argparse.ArgumentParser(description="Detekce plevele na sedem chodniku a spinani ventilu.")
    parser.add_argument("--loop-delay", type=float, default=0.08, help="Pauza mezi kroky v sekundach.")
    parser.add_argument("--hold-seconds", type=float, default=0.20, help="Jak dlouho po detekci drzet ventil otevreny.")
    parser.add_argument("--dry-run", action="store_true", help="Jen loguje, nespina rele.")
    args = parser.parse_args()

    left_detector = WeedDetector(
        min_path_ratio=SIDE_MIN_PATH_RATIO,
        min_green_ratio=SIDE_MIN_GREEN_RATIO,
        crop_top_ratio=SIDE_CROP_TOP_RATIO,
        crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
        zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
    )
    front_detector = WeedDetector(
        min_path_ratio=FRONT_MIN_PATH_RATIO,
        min_green_ratio=FRONT_MIN_GREEN_RATIO,
        crop_top_ratio=FRONT_CROP_TOP_RATIO,
        crop_bottom_ratio=FRONT_CROP_BOTTOM_RATIO,
        zone_width_ratio=FRONT_ZONE_WIDTH_RATIO,
    )
    right_detector = WeedDetector(
        min_path_ratio=SIDE_MIN_PATH_RATIO,
        min_green_ratio=SIDE_MIN_GREEN_RATIO,
        crop_top_ratio=SIDE_CROP_TOP_RATIO,
        crop_bottom_ratio=SIDE_CROP_BOTTOM_RATIO,
        zone_width_ratio=SIDE_ZONE_WIDTH_RATIO,
    )
    rele = None if args.dry_run else RelayController()

    front = None
    left = None
    right = None
    aktivni_ventil: Optional[str] = None
    aktivni_do = 0.0
    posledni_log = 0.0

    try:
        front = open_front_usb()
        left, right = open_side_cameras()
        print("Postrik bezi. Ctrl+C pro konec.")

        while True:
            now = time.monotonic()
            predni_det = front_detector.detect(front.read())
            levy_det = left_detector.detect(left.read())
            pravy_det = right_detector.detect(right.read())

            kandidati = []
            if levy_det.plevel:
                kandidati.append(("levy", levy_det.skore))
            if predni_det.plevel:
                kandidati.append(("predni", predni_det.skore))
            if pravy_det.plevel:
                kandidati.append(("pravy", pravy_det.skore))

            if kandidati:
                priorita = {"predni": 2, "levy": 1, "pravy": 1}
                aktivni_ventil, _ = max(kandidati, key=lambda polozka: (polozka[1], priorita.get(polozka[0], 0)))
                aktivni_do = now + args.hold_seconds
            elif now >= aktivni_do:
                aktivni_ventil = None

            levy_zap = aktivni_ventil == "levy" and now < aktivni_do
            predni_zap = aktivni_ventil == "predni" and now < aktivni_do
            pravy_zap = aktivni_ventil == "pravy" and now < aktivni_do

            if rele is not None:
                rele.set_valves(levy_zap, predni_zap, pravy_zap)

            if now - posledni_log >= 0.5:
                print(
                    "L "
                    f"{levy_det.skore:.4f}/{levy_det.skore_chodniku:.2f} "
                    f"F {predni_det.skore:.4f}/{predni_det.skore_chodniku:.2f} "
                    f"P {pravy_det.skore:.4f}/{pravy_det.skore_chodniku:.2f} "
                    f"-> ventil={aktivni_ventil or '-'} L={int(levy_zap)} F={int(predni_zap)} P={int(pravy_zap)}"
                )
                posledni_log = now

            time.sleep(args.loop_delay)
    except KeyboardInterrupt:
        print("Konec postriku.")
        return 0
    finally:
        if rele is not None:
            rele.close()
        for kamera in (front, left, right):
            if kamera is not None:
                try:
                    kamera.close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
