import tkinter as tk
from tkinter import messagebox

from cynes.windowed import WindowedNES
from cynes import *

from threading import Thread
from time import perf_counter

from node import *
from nn import *
from activator import *
from PIL import Image, ImageDraw, ImageChops

class App(tk.Frame):
        def __init__(self, root = None) -> None:
                super().__init__(root)
                self.root = root
                self.pack()

                # variables
                self.max_nns: tk.IntVar = tk.IntVar()

                self.neural_nets: list[NeuralNet] = []

                self.create_widgets()

        def is_num(self, n: int | float) -> bool:
                return True if (not n) or n.is_integer() else False
        
        def generate_nns(self) -> None:
                def generate() -> None:
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

                        self.generated = messagebox.showinfo(
                                'Done!',
                                f'Generated {self.max_nns.get()} Neural Net{
                                        '' if self.max_nns.get() == 1 else 's'
                                }'
                        )

                if not self.neural_nets:
                        generate()
                else:
                        self.overwrite = messagebox.askyesno(
                                'Overwriting!',
                                'By clicking "Yes", you will overwrite ALL of the neural networks in the set. Do you want to continue?',
                                icon = 'warning'
                        )

                        if self.overwrite:
                                for child in self.nn_list.winfo_children():
                                        child.destroy()

                                generate()

        def evolve(self) -> None:
                movement_input = [
                        NES_INPUT_A,
                        NES_INPUT_B,
                        NES_INPUT_UP,
                        NES_INPUT_DOWN,
                        NES_INPUT_LEFT,
                        NES_INPUT_RIGHT
                ]

                evolution_start_time: float = perf_counter()

                for neural_net in self.neural_nets:
                        nes = NES('nes_rom.nes')
                        step: int = 0

                        lives: int = nes[0x75a]
                        prev_lives: int = lives

                        timer: int = nes[0x07f8] * 100 + nes[0x07f9] * 10 + nes[0x07fa]
                        prev_timer: int = timer

                        score: int = sum(nes[2013 + i] * 100 ** (5 - i) for i in range(6))
                        prev_score: int = score

                        print(f'Evaluating Neural Network {self.neural_nets.index(neural_net)} of ID {neural_net.id}')
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

                                        if timer < prev_timer:
                                                neural_net.score -= 0.1
                                        prev_timer = timer

                                        if lives < prev_lives:
                                                neural_net.score -= 5 + 0.1 * (400 - timer)
                                        prev_lives = lives

                                        # positive
                                        if score > prev_score:
                                                neural_net.score += (score - prev_score) / 10

                                        prev_score = score

                                step += 1

                                if nes.has_crashed or nes[0x0770] == 0x3: break

                        nn_end_time: float = perf_counter()

                        print(f'Time elapsed for {self.neural_nets.index(neural_net)}: {
                                nn_end_time - nn_start_time
                        } seconds')

                evolution_end_time: float = perf_counter()

                print(f'Total elapsed time: {evolution_end_time - evolution_start_time}')

        def evolve_thread(self) -> None:
                evo_thread = Thread(target=self.evolve)
                evo_thread.start()

        def create_widgets(self) -> None:
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

                self.button_gen_nns = tk.Button(
                        self, text='Generate Neural Nets',
                        command=self.generate_nns
                ).grid(row=1, column=0)

                self.entry_gen_nns = tk.Spinbox(
                        self, textvariable=self.max_nns,
                        from_=1, to=25
                )

                self.entry_gen_nns.grid(row=1, column=1)

                self.label_do_epoch = tk.Label(
                        self, text='Do Epoch(s): '
                ).grid(row=2, column=0)

                self.entry_do_epoch = tk.Entry(
                        self, validatecommand=(self.root.register(self.is_num), '%P')
                ).grid(row=2, column=1)

                self.button_do_epoch = tk.Button(
                        self, text='Run',
                        command=self.evolve_thread
                ).grid(row=2, column=2)

                self.button_stop_epoch = tk.Button(
                        self, text='Stop'
                ).grid(row=2, column=3)
                

root = tk.Tk()
root.title('DigiPlay Control Panel')
root.minsize(600, 350)
root.maxsize(1000, 600)
app = App(root)
app.mainloop()