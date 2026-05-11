import tkinter as tk
from tkinter import messagebox

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
                                        '' if self.max_nns.get() == 0 else 's'
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
                        self, text='Run'
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