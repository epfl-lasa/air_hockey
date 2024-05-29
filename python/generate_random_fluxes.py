
import os
import random
import matplotlib.pyplot as plt

def generate_gaussian_values_file(filename, num_values, mean, stddev):
    with open(filename, 'w') as file:
        for _ in range(num_values):
            random_value = max(0.5, min(0.9, random.gauss(mean, stddev)))
            file.write(f"{random_value:.2f}\n")

def generate_random_values_file(filename, num_values):
    with open(filename, 'w') as file:
        for _ in range(num_values):
            random_value = random.uniform(0.5, 0.9)
            file.write(f"{random_value:.3f}\n")

def sort_file(input_file, output_file, remove_input=False, reverse=False):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    if remove_input:
        os.remove(input_file)

    sorted_lines = sorted(lines,reverse=reverse)

    with open(output_file, 'w') as file:
        for line in sorted_lines:
            file.write(line)

def plot_values(filename):
    with open(filename, 'r') as file:
        values = [float(line.strip()) for line in file]

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=70, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Generated Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    ## SET FILE PARAMETERS 
    folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/air_hockey/desired_hitting_fluxes"
    
    filename = "random_fluxes_uni_300_to_09"  

    # SET GENERATION PARAMETERS 
    num_values = 300  # Replace with the number of random values you want
    mean = 0.85  # Adjust the mean according to your preference
    stddev = 0.15 # Adjust the standard deviation according to your preference

    complete_path = os.path.join(folder,filename+".txt")
    complete_path_sorted = os.path.join(folder,filename+"_sorted.txt")
    
    # generate_gaussian_values_file(complete_path, num_values, mean, stddev)
    generate_random_values_file(complete_path, num_values)
    sort_file(complete_path, complete_path_sorted, remove_input=True, reverse = True)
    plot_values(complete_path_sorted)
    