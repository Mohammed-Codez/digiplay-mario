import tkinter as tk
from tkinter import messagebox

from cynes.windowed import WindowedNES
from cynes import *

from threading import Thread
from time import perf_counter

import os, shutil
import platform

import numpy as np
from node import *
from nn import *
from activator import *
from PIL import Image, ImageDraw, ImageChops

class App(tk.Frame):
        def __init__(self, root: tk.Tk | None = None) -> None:
                super().__init__(root)
                self.root = root
                self.pack()

                # variables
                self.max_nns: tk.IntVar = tk.IntVar()
                self.epochs: tk.IntVar = tk.IntVar()
                self.generations: int = 0

                self.weight_mutation_rate: tk.DoubleVar = tk.DoubleVar(value = 10)
                self.bias_mutation_rate: tk.DoubleVar = tk.DoubleVar(value = 10)

                self.weight_mutation_scale: tk.DoubleVar = tk.DoubleVar(value = 1)
                self.bias_mutation_scale: tk.DoubleVar = tk.DoubleVar(value = 1)

                self.neural_nets: list[NeuralNet] = []

                self.create_widgets()

        def is_num(self, n: int | float) -> bool:
                return True if (not n) or n.is_integer() else False
        
        def setup_nns(self) -> None:
                self.neural_nets = []
                
                for neural_net in range(self.max_nns.get()):
                        nn: NeuralNet = NeuralNet(
                                128 * 120,
                                [6],
                                id = neural_net
                        )

                        nn.generate_random_weights()
                        nn.generate_random_biases(0)

                        self.neural_nets.append(nn)

                        frame_child = tk.Button(
                                self.nn_list,
                                text=neural_net
                        )

                        frame_child.pack()

        def update_nns(self) -> None:
                for child in self.nn_list.winfo_children():
                        child.destroy()

                self.setup_nns()
        
        def generate_nns(self) -> None:
                if not self.neural_nets:
                        self.setup_nns()
                else:
                        self.overwrite = messagebox.askyesno(
                                'Overwriting!',
                                'By clicking "Yes", you will overwrite ALL of the neural networks in the set. Do you want to continue?',
                                icon = 'warning'
                        )

                        if self.overwrite:
                                self.update_nns()

        def evaluate(self, neural_net: NeuralNet) -> None:
                movement_input = [
                        NES_INPUT_A,
                        NES_INPUT_B,
                        NES_INPUT_UP,
                        NES_INPUT_DOWN,
                        NES_INPUT_LEFT,
                        NES_INPUT_RIGHT
                ]

                evaluation_start_time: float = perf_counter()

                # for neural_net in self.neural_nets:
                nes = NES('nes_rom.nes')
                step: int = 0

                lives: int = nes[0x75a]
                prev_lives: int = lives

                timer: int = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]
                prev_timer: int = timer

                score: int = sum(nes[2013 + i] * 100 ** (5 - i) for i in range(6))
                prev_score: int = score

                neural_net.score = 0

                print(f'Evaluating Neural Network {self.neural_nets.index(neural_net) + 1} of ID {neural_net.id}...')
                nn_start_time: float = perf_counter()

                while not nes.has_crashed or nes[0x0770] != 0x3: # if game over
                        frame = nes.step()

                        lives = nes[0x75a]
                        timer = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]
                        score = sum(nes[2013 + i] * 100 ** (5 - i) for i in range(6))

                        if step == 30: nes.controller = NES_INPUT_START

                        if step >= 40:
                                img = Image.fromarray(frame).resize((128, 120)).convert('L')
                                nn_input = np.asarray(img).ravel().tolist()

                                calced_movements = neural_net.calc(nn_input)

                                for i in range(6):
                                        nes.controller |= bool(calced_movements[i]) and bool(movement_input[i])

                                # scoring
                                if nes[0x0490] == 0xfe: # if player is colliding
                                        neural_net.score -= 5 / 30
                                        print(f'Neural Net {self.neural_nets.index(neural_net)} hit wall. Score: {neural_net.score}')

                                if timer < prev_timer:
                                        neural_net.score -= 0.1
                                        print(f'Neural Net {self.neural_nets.index(neural_net)} wasted time to {timer}. Score: {neural_net.score}')

                                prev_timer = timer

                                if lives < prev_lives:
                                        neural_net.score -= 5 + 0.1 * (400 - timer)
                                        print(f'Neural Net {self.neural_nets.index(neural_net)} lost life to {lives + 1}. Score: {neural_net.score}')

                                prev_lives = lives

                                # positive
                                if score > prev_score:
                                        neural_net.score += (score - prev_score) / 10
                                        print(f'Neural Net {self.neural_nets.index(neural_net)} gained points by {score - prev_score} to {score}. Score: {neural_net.score}')

                                prev_score = score

                        step += 1

                        if nes[0x0770] == 0x3:
                                neural_net.score -= 20
                                print(f'Neural Net {self.neural_nets.index(neural_net)} ended game. Score: {neural_net.score}')

                        if nes.has_crashed or nes[0x0770] == 0x3: break

                nn_end_time: float = perf_counter()
                print(f'Time elapsed for {self.neural_nets.index(neural_net)}: {nn_end_time - nn_start_time} seconds')

                evaluation_end_time: float = perf_counter()
                # print(f'Total elapsed time: {evaluation_end_time - evaluation_start_time}')

        def evaluate_thread(self) -> None:
                eval_threads: list[Thread] = []

                for neural_net in self.neural_nets:
                        eval_thread = Thread(target=self.evaluate, args=(neural_net,))
                        eval_thread.start()
                        eval_threads.append(eval_thread)

                for eval_thread in eval_threads:
                        eval_thread.join()

        def kill_and_reproduce(self) -> None:
                new_nns: list[NeuralNet] = self.neural_nets.copy()

                new_nns.sort(key=lambda nn: nn.score, reverse=True)

                while len(new_nns) > np.ceil(self.max_nns.get() / 2):
                        new_nns.pop()
                
                halved_nns: list[NeuralNet] = new_nns.copy()
                # print(len(halved_nns))

                reproduction_start_time: float = perf_counter()

                for neural_net in halved_nns:
                        if len(new_nns) < self.max_nns.get():
                                for layer in neural_net.layers:
                                        for node in layer.nodes:
                                                if np.random.uniform(0, 1) < (self.weight_mutation_rate.get() / 100):
                                                        node.weights += np.random.normal(scale=self.weight_mutation_scale.get(), size=node.weights.size)
                                                        print(f'Modified node weights in Layer {neural_net.layers.index(layer)} of NN {self.neural_nets.index(neural_net)}')

                                                if np.random.uniform(0, 1) < (self.bias_mutation_rate.get() / 100):
                                                        node.bias += np.random.normal(scale=self.bias_mutation_scale.get())
                                                        print(f'Modified node bias in Layer {neural_net.layers.index(layer)} of NN {self.neural_nets.index(neural_net)}')

                                        if np.random.uniform(0, 1) < 0.02:
                                                layer.nodes.append(Node(
                                                        layer.input_count + 1,
                                                        np.random.rand(layer.input_count + 1).tolist(),
                                                        np.random.uniform(-1, 1),
                                                        layer.activator
                                                ))

                                                print(f'Added node in Layer {neural_net.layers.index(layer)} of NN {self.neural_nets.index(neural_net)}')

                                if np.random.uniform(0, 1) < 0.02:
                                        new_layer_pos: int = np.random.randint(0, len(neural_net.layers) - 1)
                                        new_layer_inp_size: int = 128*120 if new_layer_pos == 0 else neural_net.layer_sizes[new_layer_pos - 1]

                                        neural_net.layers.insert(new_layer_pos, Layer(
                                                new_layer_inp_size,
                                                1,
                                                np.random.rand(1, new_layer_inp_size).tolist(),
                                                np.random.rand(new_layer_inp_size).tolist(),
                                                silu
                                        ))

                                        print(f'Added new layer at {new_layer_pos} in NN {self.neural_nets.index(neural_net)}')

                        new_nns.append(neural_net)

                self.neural_nets = new_nns.copy()
                
                reproduction_end_time: float = perf_counter()
                print(f'Time elapsed for reproduction: {reproduction_end_time - reproduction_start_time} seconds')

                self.update_nns()

        def evolve(self) -> None:
                for epoch in range(self.epochs.get()):
                        self.generations += 1
                        print(f'Generation {self.generations}:')
                        tk.Label(
                                self, text=f'Generation: {self.generations}'
                        ).grid(row=5, column=0)
                        print('Evaluating...')
                        self.evaluate_thread()
                        print('Evaluation complete!')
                        print('Evolving...')
                        self.kill_and_reproduce()
                        print('Evolution complete!')

                print('Done evolving!')

        def evolve_thread(self) -> None:
                evo_thread = Thread(target=self.evolve)
                evo_thread.start()

        def save_nns(self) -> None:
                if not os.path.exists('saved_nns'):
                        os.makedirs('saved_nns')

                for file in os.listdir('saved_nns'):
                        file_path = os.path.join('saved_nns', file)
                        try:
                                if os.path.isfile(file_path) or os.path.islink(file_path):
                                        os.remove(file_path)
                                elif os.path.isdir(file_path):
                                        shutil.rmtree(file_path)
                        except Exception as e:
                                print(f'Failed to delete {file_path}. Reason: {e}')

                for neural_net in self.neural_nets:
                        np.savez(
                                f'saved_nns/nn_{neural_net.id}.npz', 
                                layer_sizes=neural_net.layer_sizes,
                                weights=[layer.weights for layer in neural_net.layers],
                                biases=[layer.biases for layer in neural_net.layers]
                        )

        def load_nns(self) -> None:
                for child in self.nn_list.winfo_children():
                        child.destroy()

                self.neural_nets = []

                for file in os.listdir('saved_nns'):
                        if file.endswith('.npz'):
                                data = np.load(f'saved_nns/{file}', allow_pickle=True)
                                nn = NeuralNet(
                                        128 * 120,
                                        data['layer_sizes'].tolist(),
                                        weights=data['weights'].tolist(),
                                        biases=data['biases'].tolist(),
                                        id=int(file.split('_')[1].split('.')[0])
                                )

                                self.neural_nets.append(nn)

                                frame_child = tk.Button(
                                        self.nn_list,
                                        text=self.neural_nets.index(nn)
                                )

                                frame_child.pack()



        def create_widgets(self) -> None:
                self.menu = tk.Menu(self)
        
                self.menu_file = tk.Menu(self.menu, tearoff=0)
                
                self.menu_file.add_command(
                        label='Save Neural Nets',
                        command=self.save_nns,
                        accelerator='Cmd-S' if platform.system() == 'Darwin' else 'Ctrl-S',
                )

                self.menu_file.add_command(
                        label='Load Neural Nets',
                        command=self.load_nns,
                        accelerator='Cmd-O' if platform.system() == 'Darwin' else 'Ctrl-O',
                )

                self.menu.add_cascade(label='File', menu=self.menu_file)

                # ================

                self.nn_list = tk.Scrollbar(
                        self,
                        width=(root.winfo_width()),
                        bd=1, bg='white',
                )

                self.nn_list.grid(
                        row=0, column=0,
                        columnspan=2
                )

                self.nn_visualizer = tk.Canvas(
                        self,
                        width=(root.winfo_width()), height=(root.winfo_width()),
                        bd=1, bg='white',
                )

                self.nn_visualizer.grid(
                        row=0, column=2,
                        columnspan=2
                )

                # ================

                self.button_gen_nns = tk.Button(
                        self, text='Generate Neural Nets',
                        command=self.generate_nns
                ).grid(row=1, column=0)

                self.entry_gen_nns = tk.Spinbox(
                        self, textvariable=self.max_nns,
                        from_=1, to=25
                )

                # ================

                self.entry_gen_nns.grid(row=1, column=1)

                self.label_do_epoch = tk.Label(
                        self, text='Do Epoch(s): '
                ).grid(row=2, column=0)

                self.entry_do_epoch = tk.Spinbox(
                        self, textvariable=self.epochs,
                        from_=1, to=10
                )
                
                self.entry_do_epoch.grid(row=2, column=1)

                self.button_do_epoch = tk.Button(
                        self, text='Run',
                        command=self.evolve_thread
                ).grid(row=2, column=2)

                self.button_stop_epoch = tk.Button(
                        self, text='Stop'
                ).grid(row=2, column=3)

                # ================

                self.button_weight_mutation_rate = tk.Label(
                        self, text='Weight Mutation Rate',
                ).grid(row=3, column=0)

                self.entry_weight_mutation_rate = tk.Spinbox(
                        self, textvariable=self.weight_mutation_rate,
                        from_=1, to=100
                )

                self.entry_weight_mutation_rate.grid(row=3, column=1)

                self.button_bias_mutation_rate = tk.Label(
                        self, text='Bias Mutation Rate',
                ).grid(row=3, column=2)

                self.entry_bias_mutation_rate = tk.Spinbox(
                        self, textvariable=self.bias_mutation_rate,
                        from_=1, to=100
                )

                self.entry_bias_mutation_rate.grid(row=3, column=3)

                # ================

                self.button_weight_mutation_scale = tk.Label(
                        self, text='Weight Mutation Scale',
                ).grid(row=4, column=0)

                self.entry_weight_mutation_scale = tk.Spinbox(
                        self, textvariable=self.weight_mutation_scale,
                        from_=1, to=10
                )

                self.entry_weight_mutation_scale.grid(row=4, column=1)

                self.button_bias_mutation_scale = tk.Label(
                        self, text='Bias Mutation Scale',
                ).grid(row=4, column=2)

                self.entry_bias_mutation_scale = tk.Spinbox(
                        self, textvariable=self.bias_mutation_scale,
                        from_=1, to=10
                )

                self.entry_bias_mutation_scale.grid(row=4, column=3)

                # ================

                self.button_save_nns = tk.Button(
                        self, text='Save Neural Nets',
                        command=self.save_nns
                ).grid(row=5, column=0, columnspan=2)

                self.button_load_nns = tk.Button(
                        self, text='Load Neural Nets',
                        command=self.load_nns
                ).grid(row=5, column=2, columnspan=2)

                # ================

                self.info_generation = tk.Label(
                        self, text=f'Generation: {self.generations}'
                ).grid(row=6, column=0)

                # ================

                self.root.config(menu=self.menu)
                self.root.bind_all('<Command-s>' if platform.system() == 'Darwin' else '<Control-s>', lambda event: self.save_nns())
                self.root.bind_all('<Command-o>' if platform.system() == 'Darwin' else '<Control-o>', lambda event: self.load_nns())

root = tk.Tk()
root.title('DigiPlay Control Panel')
root.minsize(600, 350)
root.maxsize(1000, 600)
app = App(root)
app.mainloop()