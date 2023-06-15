import tkinter as tk
import NeuralNetworkV4
from tkinter import ttk
import threading
import queue
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox

message_queue = []  # Store the messages
current_message_index = 0  # Index of the current message being displayed
# Set default values for the global variables
default_word_hyper_param = 1
default_letter_hyper_param = 0.5
default_pair_letters_hyper_param = 0.5
default_hyper_letter_correct = 0.5
default_hyper_pair_letters_correct = 0.5
default_mutation_rate_starting = 0.1
default_mutation_trashold = 0.005
default_increase_mutation = 0.08
default_decrease_mutation = 0.05
default_improvement_rates_queue_length = 5
default_N = 5
default_input_txt = 'enc.txt'
default_true_txt = 'true_perm.txt'

def check_queue(q):
    global current_message_index

    try:
        msg = q.get_nowait()  # try to get a message from the queue
    except queue.Empty:
        pass  # if there's nothing in the queue, do nothing
    else:
        if msg[0] == 'result':
            # If the message is a 'result' message, show the messagebox with the results
            fitness_calls, number_generations, final_fitness_score, final_solutions, accuracy = msg[1], msg[2], msg[3], msg[4], msg[5]

            messagebox.showinfo("Results", f"Fitness calls:  {fitness_calls}\n"
                                           f"Number of Generations:  {number_generations}\n"
                                           f"Fitness score:  {final_fitness_score:.3f}\n"                           
                                           f"Final solutions:  {final_solutions}\n"
                                           f"Accuracy:  {accuracy}%")
        else:
            # Otherwise, add the message to the queue as before
            message_queue.append(msg)

    if current_message_index < len(message_queue):
        output_text.insert(tk.END, message_queue[current_message_index] + '\n')  # display the current message
        output_text.see(tk.END)  # scroll to the last line
        current_message_index += 1

    if current_message_index >= len(message_queue):
        root.after(100, check_queue, q)  # wait for 5 seconds before checking the queue again
    else:
        root.after(100, check_queue, q)  # check the queue again after 1 ms

def on_close():
    print("Closing program... Thank you for checking the assigment!")
    root.destroy()
    sys.exit()

def run_genetic_algorithm():
    global current_message_index
    # Clear the figure, message queue and output text
    fig.clf()
    message_queue.clear()
    output_text.delete(1.0, tk.END)
    current_message_index = 0  # Reset the current message index

    population_size = int(population_size_entry.get())
    input_dim = float(mutation_rate_starting_entry.get())
    hidden1_dim = float(max_mutation_rate_entry.get())
    hidden2_dim = float(min_mutation_rate_entry.get())
    output_dim = int(max_iterations_entry.get())
    generations = bool(elitism_var.get())


    #GeneticAlgorithm.genetic_algorithm(cipher_text, optimization, population_size, max_mutation_rate, min_mutation_rate, max_iterations, elitism)
    q = queue.Queue()
    thread = threading.Thread(target=NeuralNetworkV4.execute, args=(q, fig, canvas, X_train, y_train, X_test, y_test, input_dim, hidden1_dim, hidden2_dim, output_dim, population_size, generations))
    thread.daemon = True
    thread.start()
    check_queue(q)  # check the queue for messages right away


root = tk.Tk()

fig = Figure(figsize=(5, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create a frame to hold the input fields
input_frame = tk.Frame(root)
input_frame.pack()





# Create input fields for the global variables with default values
word_hyper_param_label = tk.Label(input_frame, text="Word hyper parameter:")
word_hyper_param_label.grid(row=0, column=0)
word_hyper_param_entry = tk.Entry(input_frame)
word_hyper_param_entry.grid(row=0, column=1)
word_hyper_param_entry.insert(0, str(default_word_hyper_param))

letter_hyper_param_label = tk.Label(input_frame, text="Letter hyper parameter:")
letter_hyper_param_label.grid(row=1, column=0)
letter_hyper_param_entry = tk.Entry(input_frame)
letter_hyper_param_entry.grid(row=1, column=1)
letter_hyper_param_entry.insert(0, str(default_letter_hyper_param))

pair_letters_hyper_param_label = tk.Label(input_frame, text="Pair letters hyper parameter:")
pair_letters_hyper_param_label.grid(row=2, column=0)
pair_letters_hyper_param_entry = tk.Entry(input_frame)
pair_letters_hyper_param_entry.grid(row=2, column=1)
pair_letters_hyper_param_entry.insert(0, str(default_pair_letters_hyper_param))

hyper_letter_correct_label = tk.Label(input_frame, text="Hyper letter correct:")
hyper_letter_correct_label.grid(row=3, column=0)
hyper_letter_correct_entry = tk.Entry(input_frame)
hyper_letter_correct_entry.grid(row=3, column=1)
hyper_letter_correct_entry.insert(0, str(default_hyper_letter_correct))

hyper_pair_letters_correct_label = tk.Label(input_frame, text="Hyper pair letters correct:")
hyper_pair_letters_correct_label.grid(row=4, column=0)
hyper_pair_letters_correct_entry = tk.Entry(input_frame)
hyper_pair_letters_correct_entry.grid(row=4, column=1)
hyper_pair_letters_correct_entry.insert(0, str(default_hyper_pair_letters_correct))

improvement_rates_queue_length_label = tk.Label(input_frame, text="Improvement rates queue length:")
improvement_rates_queue_length_label.grid(row=4, column=2)
improvement_rates_queue_length_entry = tk.Entry(input_frame)
improvement_rates_queue_length_entry.grid(row=4, column=3)
improvement_rates_queue_length_entry.insert(0, str(default_improvement_rates_queue_length))

N_label = tk.Label(input_frame, text="N:")
N_label.grid(row=0, column=4)
N_entry = tk.Entry(input_frame)
N_entry.grid(row=0, column=5)
N_entry.insert(0, str(default_N))

population_size_label = tk.Label(input_frame, text="Population number:")
population_size_label.grid(row=1, column=4)
population_size_entry = tk.Entry(input_frame)
population_size_entry.grid(row=1, column=5)
population_size_entry.insert(0, "200")

mutation_trashold_label = tk.Label(input_frame, text="Mutation threshold:")
mutation_trashold_label.grid(row=1, column=2)
mutation_trashold_entry = tk.Entry(input_frame)
mutation_trashold_entry.grid(row=1, column=3)
mutation_trashold_entry.insert(0, str(default_mutation_trashold))

increase_mutation_label = tk.Label(input_frame, text="Increase mutation:")
increase_mutation_label.grid(row=2, column=2)
increase_mutation_entry = tk.Entry(input_frame)
increase_mutation_entry.grid(row=2, column=3)
increase_mutation_entry.insert(0, str(default_increase_mutation))

decrease_mutation_label = tk.Label(input_frame, text="Decrease mutation:")
decrease_mutation_label.grid(row=3, column=2)
decrease_mutation_entry = tk.Entry(input_frame)
decrease_mutation_entry.grid(row=3, column=3)
decrease_mutation_entry.insert(0, str(default_decrease_mutation))

mutation_rate_starting_label = tk.Label(input_frame, text="Mutation rate:")
mutation_rate_starting_label.grid(row=0, column=2)
mutation_rate_starting_entry = tk.Entry(input_frame)
mutation_rate_starting_entry.grid(row=0, column=3)
mutation_rate_starting_entry.insert(0, str(default_mutation_rate_starting))

max_mutation_rate_label = tk.Label(input_frame, text="Maximum mutation rate:")
max_mutation_rate_label.grid(row=2, column=4)
max_mutation_rate_entry = tk.Entry(input_frame)
max_mutation_rate_entry.grid(row=2, column=5)
max_mutation_rate_entry.insert(0, "0.4")

min_mutation_rate_label = tk.Label(input_frame, text="Minimum mutation rate:")
min_mutation_rate_label.grid(row=3, column=4)
min_mutation_rate_entry = tk.Entry(input_frame)
min_mutation_rate_entry.grid(row=3, column=5)
min_mutation_rate_entry.insert(0, "0.1")

max_iterations_label = tk.Label(input_frame, text="Maximum iterations:")
max_iterations_label.grid(row=4, column=4)
max_iterations_entry = tk.Entry(input_frame)
max_iterations_entry.grid(row=4, column=5)
max_iterations_entry.insert(0, "150")

elitism_var = tk.IntVar()
elitism_checkbox = tk.Checkbutton(input_frame, text="Elitism", variable=elitism_var)
elitism_checkbox.grid(row=5, column=5)
elitism_var.set(1)  # Default value for elitism is set to True (1)

# random_mutation_func_var = tk.IntVar()
# random_mutation_func_checkbox = tk.Checkbutton(input_frame, text="Use PURE random mutation function", variable=random_mutation_func_var)
# random_mutation_func_checkbox.grid(row=5, column=0)  # Put this under Elitism checkbox
# random_mutation_func_var.set(0)  # Default value for random_mutation_func is set to False (0)

file_name_label = tk.Label(input_frame, text="Encrypt File Name:")
file_name_label.grid(row=6, column=4)
file_name_label.config(font=("TkDefaultFont",10, "bold"))  # Set label font to bold
file_name_entry = tk.Entry(input_frame)
file_name_entry.grid(row=6, column=5)
file_name_entry.insert(0, default_input_txt)

true_permutation_label = tk.Label(input_frame, text="True Permutation File Name:")
true_permutation_label.grid(row=6, column=0)
true_permutation_label.config(font=("TkDefaultFont", 10, "bold"))  # Set label font to bold
true_permutation_entry = tk.Entry(input_frame)
true_permutation_entry.grid(row=6, column=1)
true_permutation_entry.insert(0, default_true_txt)

# Create the combobox for selecting the optimization strategy
optimization_label = tk.Label(root, text="Optimization strategy:")
optimization_label.config(font=("TkDefaultFont", 10, "bold"))  # Set label font to bold
optimization_label.pack()
optimization_combobox = ttk.Combobox(root, values=["None", "Darwinian", "Lamarckian"])
optimization_combobox.pack()
optimization_combobox.current(0)  # set initial selection to "None"

# Set real only state for the parameters so the user cant change them.
word_hyper_param_entry.configure(state='readonly')
letter_hyper_param_entry.configure(state='readonly')
pair_letters_hyper_param_entry.configure(state='readonly')
hyper_letter_correct_entry.configure(state='readonly')
hyper_pair_letters_correct_entry.configure(state='readonly')
mutation_trashold_entry.configure(state='readonly')
increase_mutation_entry.configure(state='readonly')
decrease_mutation_entry.configure(state='readonly')
improvement_rates_queue_length_entry.configure(state='readonly')
N_entry.configure(state='readonly')
population_size_entry.configure(state='readonly')
mutation_rate_starting_entry.configure(state='readonly')
max_mutation_rate_entry.configure(state='readonly')
min_mutation_rate_entry.configure(state='readonly')
max_iterations_entry.configure(state='readonly')
elitism_checkbox.configure(state='disabled')

run_button = tk.Button(root, text="Run", command=run_genetic_algorithm)
run_button.pack()
# We choose the width by the amount of characters.
output_text = tk.Text(root, width=130)
output_text.pack()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
