import os
from dataclasses import dataclass

from gpiozero import PWMOutputDevice, DigitalOutputDevice


def _vezmi_int(klic: str, vychozi: int) -> int:
    try:
        return int(os.getenv(klic, vychozi))
    except (TypeError, ValueError):
        return vychozi


PWM_FREKVENCE = _vezmi_int("PWM_FREKVENCE", 1000)


@dataclass(frozen=True)
class PinyMotoru:
    pwm: int
    in1: int
    in2: int


MotorPins = PinyMotoru

LEFT_DEFAULT = PinyMotoru(12, 5, 6)
RIGHT_DEFAULT = PinyMotoru(13, 19, 26)


def pins_from_env(prefix: str, fallback: PinyMotoru) -> PinyMotoru:
    return PinyMotoru(
        pwm=_vezmi_int(f"{prefix}_PWM", fallback.pwm),
        in1=_vezmi_int(f"{prefix}_IN1", fallback.in1),
        in2=_vezmi_int(f"{prefix}_IN2", fallback.in2),
    )


class L298NMotor:
    def __init__(self, piny: PinyMotoru, frekvence: int = PWM_FREKVENCE):
        self.pwm = PWMOutputDevice(piny.pwm, frequency=frekvence, initial_value=0)
        self.in1 = DigitalOutputDevice(piny.in1, initial_value=False)
        self.in2 = DigitalOutputDevice(piny.in2, initial_value=False)
        self._smer = 0

    def _brzda(self) -> None:
        self.in1.on()
        self.in2.on()
        self.pwm.value = 1.0
        self._smer = 0

    def _volnobeh(self) -> None:
        self.pwm.value = 0.0
        self.in1.off()
        self.in2.off()
        self._smer = 0

    def nastav(self, rychlost: float, brzdit_pri_nule: bool = True) -> None:
        rych = max(-1.0, min(1.0, float(rychlost)))

        if rych == 0.0:
            if brzdit_pri_nule:
                self._brzda()
            else:
                self._volnobeh()
            return

        novy_smer = 1 if rych > 0 else -1
        if novy_smer != self._smer and self._smer != 0:
            self._brzda()

        if novy_smer > 0:
            self.in1.on()
            self.in2.off()
        else:
            self.in1.off()
            self.in2.on()

        self.pwm.value = abs(rych)
        self._smer = novy_smer

    def zastav(self, brzdit: bool = True) -> None:
        if brzdit:
            self._brzda()
        else:
            self._volnobeh()

    def close(self) -> None:
        try:
            self._volnobeh()
        except Exception:
            pass
        for zarizeni in (self.pwm, self.in1, self.in2):
            try:
                zarizeni.close()
            except Exception:
                pass


class L298NDriver:
    def __init__(self, levy: PinyMotoru, pravy: PinyMotoru, frekvence: int = PWM_FREKVENCE):
        self.levy_motor = L298NMotor(levy, frekvence)
        self.pravy_motor = L298NMotor(pravy, frekvence)

    def nastav(self, levy_rychlost: float, pravy_rychlost: float) -> None:
        self.levy_motor.nastav(levy_rychlost)
        self.pravy_motor.nastav(pravy_rychlost)

    def zastav(self, brzdit: bool = True) -> None:
        self.levy_motor.zastav(brzdit)
        self.pravy_motor.zastav(brzdit)

    def close(self) -> None:
        self.levy_motor.close()
        self.pravy_motor.close()
