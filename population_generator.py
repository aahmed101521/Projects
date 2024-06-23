import numpy as np
import matplotlib.pyplot as plt

class PopulationGenerator:
    def __init__(self, size = 10000):
        """
        
        Initialize the population generator
        
        Args:
        size (int): The size of the popuzlation to generate
        """
        self.size = size
        self.population = None
        
    def generate_normal(self, mean = 0, std =1):
        """
        
        Generate a normally distributed population.
        
        Args:
        mean(float): Mean of the distribution
        std(float): Standard deviation
        """    
        self.population = np.random.normal(mean, std, self.size)
        
    def generate_uniform(self, low=0, high=1):
        """
        Generate a uniformly distributed population.
        
        Args:
        low (float): Lower bound of the distribution
        high (float): Upper bound of the distribution
        """
        self.population = np.random.uniform(low, high, self.size)

    def generate_exponential(self, scale=1.0):
        """
        Generate an exponentially distributed population.
        
        Args:
        scale (float): The scale parameter of the exponential distribution
        """
        self.population = np.random.exponential(scale, self.size)

    def generate_bimodal(self, mean1=0, std1=1, mean2=5, std2=1, ratio=0.5):
        """
        Generate a bimodal population (mixture of two normal distributions).
        
        Args:
        mean1, std1 (float): Mean and std dev of the first distribution
        mean2, std2 (float): Mean and std dev of the second distribution
        ratio (float): Ratio of the first distribution in the mixture (0 to 1)
        """
        size1 = int(self.size * ratio)
        size2 = self.size - size1
        pop1 = np.random.normal(mean1, std1, size1)
        pop2 = np.random.normal(mean2, std2, size2)
        self.population = np.concatenate([pop1, pop2])    
    def visualize_population(self):
        """
        Visualize the generated population.
        """
        if self.population is None:
            print("No population generated yet.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.population, bins=50, edgecolor='black')
        plt.title("Population Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    
    def get_stats(self):
        """
        Get basic statistics of the population.
        """
        if self.population is None:
            return "No population generated yet."

        return {
            "mean": np.mean(self.population),
            "median": np.median(self.population),
            "std": np.std(self.population),
            "min": np.min(self.population),
            "max": np.max(self.population)
        }
        
# Usage example
if __name__ == "__main__":
    # Create a population generator
    gen = PopulationGenerator(size=100000)

    # Generate different types of populations and visualize them
    print("Normal Distribution:")
    gen.generate_normal(mean=0, std=1)
    gen.visualize_population()
    print(gen.get_stats())

    print("\nUniform Distribution:")
    gen.generate_uniform(low=-3, high=3)
    gen.visualize_population()
    print(gen.get_stats())

    print("\nExponential Distribution:")
    gen.generate_exponential(scale=2)
    gen.visualize_population()
    print(gen.get_stats())

    print("\nBimodal Distribution:")
    gen.generate_bimodal(mean1=-2, std1=1, mean2=2, std2=1, ratio=0.6)
    gen.visualize_population()
    print(gen.get_stats())                