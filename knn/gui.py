import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import main


class Gui:
    def __init__(self):
        self.root = root
        self.root.title("k-NN Classifier")
        self.root.geometry("800x600")

        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20)

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20)

        self.mode_label = tk.Label(self.left_frame, text="Choose an operation:")
        self.mode_label.pack()

        self.mode_var = tk.StringVar(value="classify")
        self.classify_radio = tk.Radiobutton(self.left_frame, text="Classify Vector", variable=self.mode_var,
                                             value="classify", command=self.update_ui)
        self.classify_radio.pack()

        self.accuracy_radio = tk.Radiobutton(self.left_frame, text="Calculate Accuracy", variable=self.mode_var,
                                             value="accuracy", command=self.update_ui)
        self.accuracy_radio.pack()

        self.k_label = tk.Label(self.left_frame, text="Enter number of neighbors (k):")
        self.k_label.pack()
        self.k_entry = tk.Entry(self.left_frame)
        self.k_entry.pack()

        self.vector_label = tk.Label(self.left_frame, text="Enter vector to classify (comma separated):")
        self.vector_label.pack()
        self.vector_entry = tk.Entry(self.left_frame)
        self.vector_entry.pack()

        self.action_button = tk.Button(self.left_frame, text="Run", command=self.run_action)
        self.action_button.pack(pady=5)

        self.result_label = tk.Label(self.left_frame, text="")
        self.result_label.pack()

        self.train_set = 'Train-set.csv'
        self.test_set = 'Test-set.csv'

        self.canvas = None

        self.plot_data()

    def update_ui(self):
        mode = self.mode_var.get()
        if mode == "classify":
            self.vector_label.pack()
            self.vector_entry.pack()
        else:
            self.vector_label.pack_forget()
            self.vector_entry.pack_forget()

    def run_action(self):
        mode = self.mode_var.get()
        try:
            k = int(self.k_entry.get())

            if mode == "classify":
                vector_input = self.vector_entry.get()
                vector = [float(i) for i in vector_input.split(",")]
                predicted_class = main.single(self.train_set, vector, k)
                self.result_label.config(text=f"Predicted class: {predicted_class}")
            elif mode == "accuracy":
                acc, _ = main.knn(self.train_set, self.test_set, k)
                self.result_label.config(text=f"Accuracy: {acc:.2f}%")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid vector and number of neighbors.")

    def plot_data(self):
        train = main.load_csv(self.train_set)

        setosa = [row for row in train if row[-1] == 'Iris-setosa']
        versicolor = [row for row in train if row[-1] == 'Iris-versicolor']
        virginica = [row for row in train if row[-1] == 'Iris-virginica']

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([float(row[0]) for row in setosa], [float(row[1]) for row in setosa], color='red', label='Setosa')
        ax.scatter([float(row[0]) for row in versicolor], [float(row[1]) for row in versicolor], color='blue',
                   label='Versicolor')
        ax.scatter([float(row[0]) for row in virginica], [float(row[1]) for row in virginica], color='green',
                   label='Virginica')

        ax.set_xlabel("Sepal length")
        ax.set_ylabel("Sepal width")
        ax.set_title("Iris Data Visualization")
        ax.legend()

        if self.canvas is not None:
            self.canvas.get_tk_widget().pack_forget()

        self.canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = Gui()
    root.mainloop()
