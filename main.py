from cynes.windowed import WindowedNES
from cynes import *
from time import sleep

from nn import *
from activator import *
from PIL import Image, ImageDraw, ImageChops
import os

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

movement_list: list = [
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_A,
    NES_INPUT_B,
]

neural_nets: list[NeuralNet] = []

for i in range(10):
    nn: NeuralNet = NeuralNet(
        128*120,
        [6],
    )

    nn.layers[-1].activator = heaviside
    nn.generate_random_weights()
    nn.generate_random_biases()

    neural_nets.append(nn)

for neural_net in neural_nets:
    print(f'Recording {neural_nets.index(neural_net) + 1}')
    # We initialize a new emulator by specifying the ROM file used
    nes = NES("nes_rom.nes")

    step: int = 0
    # While the emulator should not be closed, we can continue the emulation
    while not (nes.has_crashed or nes[0x0770] == 0x3):
        # It also returns the content of the frame buffer as a numpy array
        frame = nes.step()

        if step == 30:
            nes.controller = NES_INPUT_START
            nes.step()

        if step >= 40:
            img: Image.Image = Image.fromarray(frame).resize((128, 120)).convert('L')
            nn_image: list[float] = np.asarray(img).ravel().tolist()

            calced_movements = neural_net.calc(nn_image)

            """
            # the code below might be used later

            is_nullinp = False

            draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
            draw.rectangle([
                12, 12, 116, 16
            ], '#000000')

            draw.rectangle([
                68, 40, 80, 60
            ], '#000000')
            
            for nullinp_img in os.listdir('./nullinputs'):
                img_diff = ImageChops.difference(img, Image.open(f'nullinputs/{nullinp_img}'))

                print(img_diff.getbbox())

                if not img_diff.getbbox():
                    is_nullinp = True
            """

            # if not is_nullinp:
            for i in range(len(calced_movements)):
                nes.controller |= bool(movement_list[i]) and bool(calced_movements[i])

            """
            the below code is for generating transition-frame.png
            
            if step == 40:
                draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)

                draw.rectangle([
                    12, 12, 116, 16
                ], '#000000')

                draw.rectangle([
                    68, 40, 80, 60
                ], '#000000')

                img.save('transition-frame.png')
            """

        

        step += 1
        # sleep(1 / 60)