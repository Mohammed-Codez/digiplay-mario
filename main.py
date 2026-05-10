from cynes.windowed import WindowedNES
from cynes import *
from time import sleep

from node import *
from nn import *
from activator import *
from PIL import Image, ImageDraw, ImageChops
import os
from random import randint

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
    NES_INPUT_START,
    NES_INPUT_SELECT
]

neural_nets: list[NeuralNet] = []

for i in range(10):
    layer_sizes = [
        # randint(1, 10) for _ in range(randint(2, 10))
    ]
    layer_sizes.append(6)

    nn: NeuralNet = NeuralNet(
        128*120,
        layer_sizes,
        id = i
    )

    nn.layers[-1].activator = sigmoid
    nn.generate_random_weights()
    nn.generate_random_biases(0, 1)

    neural_nets.append(nn)

for i in range(1):
    print(f'Generation {i + 1}')
    for neural_net in neural_nets:
        print(f'Running NN "{neural_net.id}"')
        # We initialize a new emulator by specifying the ROM file used
        # nes = WindowedNES("nes_rom.nes", default_handlers=False)
        nes = NES('nes_rom.nes')
        step: int = 0

        lives: int = nes[0x75a]
        prev_lives: int = nes[0x75a]

        timer: int = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]
        prev_timer: int = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]

        # While the emulator should not be closed, we can continue the emulation
        while not (nes.has_crashed or nes[0x0770] == 0x3): # if 0x0770 is 0x3, it means game over
            # It also returns the content of the frame buffer as a numpy array
            frame = nes.step()

            lives = nes[0x75a]
            timer = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]

            if step == 30:
                nes.controller = NES_INPUT_START
                nes.step()

            if step >= 40:
                img: Image.Image = Image.fromarray(frame).resize((128, 120)).convert('L')
                nn_image: list[float] = np.asarray(img).ravel().tolist()

                calced_movements = np.append(neural_net.calc(nn_image), [np.float64(0.0), np.float64(0.0)])

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
                    nes.controller |= bool(movement_list[i]) and bool(calced_movements[i] > 0.1)

                # score change
                if nes[0x0490] == 0xfe: # if player is colliding
                    neural_net.score -= 5 / 30

                if lives < prev_lives:
                    neural_net.score -= 5 + 0.1 * (400 - timer)

                prev_lives = nes[0x75a]

                if timer < prev_timer:
                    neural_net.score -= 0.1

                prev_timer: int = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]

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

            if nes.has_crashed or nes[0x0770] == 0x3:
                break

        try:
            nes.close()
        except:
            pass

    neural_nets.sort(reverse=True, key=lambda nn: nn.score)
    nn_scores: list[float] = []
    for neural_net in neural_nets:
        print(f'NN "{neural_net.id}": {neural_net.score}')
        nn_scores.append(neural_net.score)

    new_nns: list[NeuralNet] = neural_nets.copy()
    while len(new_nns) > len(neural_nets) / 2:
        new_nns.pop()

    halved_nns = new_nns.copy()

    for neural_net in halved_nns:
        if len(new_nns) >= len(neural_nets):
            break
        else:
            for layer in neural_net.layers:
                for node in layer.nodes:
                    if np.random.uniform(0, 1) < 0.05:
                        node.weights = np.array([weight + np.random.normal() for weight in node.weights])

                    if np.random.uniform(0, 1) < 0.05:
                        node.bias += np.random.normal()

                if np.random.uniform(0, 1) < 0.01:
                    layer.nodes.append(Node(
                        layer.input_count + 1,
                        np.random.rand(layer.input_count + 1).tolist(),
                        np.random.uniform(-1, 1),
                        layer.activator
                    ))

                    for node in layer.nodes:
                        node.input_count += 1 

            if np.random.uniform(0, 1) < 0.01:
                new_layer_pos: int = np.random.randint(0, len(neural_net.layers) - 1)
                new_layer_inp_size: int = 128*120 if new_layer_pos == 0 else neural_net.layer_sizes[new_layer_pos - 1]

                neural_net.layers.insert(new_layer_pos, Layer(
                    new_layer_inp_size,
                    1,
                    np.random.rand(1, new_layer_inp_size),
                    np.random.rand(new_layer_inp_size),
                    silu
                )) 

                neural_net.layer_sizes.insert(new_layer_pos, 1) 


            new_nns.append(neural_net)

    neural_nets = new_nns.copy()

    print(neural_nets)
