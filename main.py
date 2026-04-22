from cynes.windowed import WindowedNES
from cynes import *
from time import sleep
import keyboard
import pyautogui

# the below dict is a whatnot. not used. not done. idrk
keys = {
    'w' : 'up',
    'a' : 'left',
    's' : 'down',
    'd' : 'right',
    'p' : 'x',
    'l' : 'z',
    'lshift' : 'a',
    'enter' : 's'
}

# We initialize a new emulator by specifying the ROM file used
with WindowedNES("nes_rom.nes") as nes:
    # While the emulator should not be closed, we can continue the emulation
    while not nes.should_close:
        # It also returns the content of the frame buffer as a numpy array
        frame = nes.step()
        sleep(1 / 100)
