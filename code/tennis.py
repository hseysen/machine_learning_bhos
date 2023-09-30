import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk


data_path = r"..\data\tennis.csv"


class BayesianLearningApp:
    def __init__(self, root):
        """
        Initialize the app.
        """
        self.root = root
        self.root.title("Bayesian Learning Example App")
        
        self.load_data()
        self.precalculate_probabilities()
        self.create_widgets()

    def load_data(self):
        """
        Load the data.
        """
        self.dataset = pd.read_csv(data_path)

    def precalculate_probabilities(self):
        """
        Precalculate the probabilities P(X_i | C) for all columns of the data.
        Once done, the resulting attribute probabilities will be a dictionary consisting of keys as the column names, and values as the P(X_i = x | C) for all unique x in individual columns.
        """
        self.probabilities = {}
        play_yes = self.dataset["Play"] == "Yes"
        play_no = self.dataset["Play"] == "No"
        total_yes = len(self.dataset[play_yes])
        total_no = len(self.dataset[play_no])

        for i in range(len(self.dataset.columns) - 1):
            col = self.dataset.columns[i]
            unique_vals = self.dataset[col].unique()
            n_unique = len(unique_vals)

            # 2 columns for YES and NO, and n_unique rows for all unique x in the current column.
            self.probabilities[col] = np.zeros((n_unique, 2))
            for i in range(n_unique):
                self.probabilities[col][i, 0] = len((self.dataset[(self.dataset[col] == unique_vals[i]) & (play_yes)])) / total_yes
                self.probabilities[col][i, 1] = len((self.dataset[(self.dataset[col] == unique_vals[i]) & (play_no)])) / total_no

    def create_widgets(self):
        """
        Create the widgets for user input.
        """
        self.dropdown1 = ttk.Combobox(self.root, values=["Sunny", "Overcast", "Rain"])
        self.dropdown2 = ttk.Combobox(self.root, values=["Hot", "Mild", "Cool"])
        self.dropdown3 = ttk.Combobox(self.root, values=["High", "Normal"])
        self.dropdown4 = ttk.Combobox(self.root, values=["Strong", "Weak"])
        
        self.dropdown1.set("Select Outlook")
        self.dropdown2.set("Select Temperature")
        self.dropdown3.set("Select Humidity")
        self.dropdown4.set("Select Wind")

        self.button = tk.Button(self.root, text="Calculate Probability of Playing Tennis", command=self.calculate_playtennis)
        self.output_label = tk.Label(self.root, text="")

        self.dropdown1.grid(row=0, column=0, padx=10, pady=10)
        self.dropdown2.grid(row=1, column=0, padx=10, pady=10)
        self.dropdown3.grid(row=0, column=1, padx=10, pady=10)
        self.dropdown4.grid(row=1, column=1, padx=10, pady=10)
        self.button.grid(row=2, columnspan=2, padx=10, pady=10)
        self.output_label.grid(row=3, columnspan=2, padx=10, pady=10)

    def calculate_playtennis(self):
        """
        Callback function for the button on the UI. Should calculate the probability of the given inputs and update the text on the UI.
        """
        values = [self.dropdown1.get(),
                  self.dropdown2.get(),
                  self.dropdown3.get(),
                  self.dropdown4.get()]

        pyes = 1
        pno = 1
        for i in range(len(self.dataset.columns) - 1):
            col = self.dataset.columns[i]
            unique_vals = self.dataset[col].unique()
            pyes *= self.probabilities[col][np.where(unique_vals == values[i]), 0][0, 0]        # [0, 0] because np.where returns this as matrix of indices.
            pno *= self.probabilities[col][np.where(unique_vals == values[i]), 1][0, 0]         # Just makes the output cleaner.

        self.output_label.config(text=f"P(Yes) = {pyes}\tP(No) = {pno}")



# Running the file
if __name__ == "__main__":
    root = tk.Tk()
    app = BayesianLearningApp(root)
    root.mainloop()
