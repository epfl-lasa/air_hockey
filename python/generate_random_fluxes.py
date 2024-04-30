
import os
import random
import matplotlib.pyplot as plt

def generate_gaussian_values_file(filename, num_values, mean, stddev):
    with open(filename, 'w') as file:
        for _ in range(num_values):
            random_value = max(0.5, min(1.2, random.gauss(mean, stddev)))
            file.write(f"{random_value:.2f}\n")

def generate_random_values_file(filename, num_values):
    with open(filename, 'w') as file:
        for _ in range(num_values):
            random_value = random.uniform(0.5, 1.2)
            file.write(f"{random_value:.2f}\n")

def plot_values(filename):
    with open(filename, 'r') as file:
        values = [float(line.strip()) for line in file]

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Generated Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/air_hockey/desired_hitting_fluxes"
    filename = "random_fluxes_uni.txt"  
    complete_path = os.path.join(folder,filename)
    
    num_values = 300  # Replace with the number of random values you want
    mean = 0.85  # Adjust the mean according to your preference
    stddev = 0.15 # Adjust the standard deviation according to your preference

    # generate_gaussian_values_file(complete_path, num_values, mean, stddev)
    generate_random_values_file(complete_path, num_values)
    plot_values(complete_path)
    