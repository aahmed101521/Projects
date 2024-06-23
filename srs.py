import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from population_generator import PopulationGenerator

class SimpleRandomSampling:
    def __init__(self, population):
        self.population = population
        self.N = len(population)

    def select_sample(self, n, replace=False):
        """
        Select a simple random sample from the population.
        
        Args:
        n (int): Sample size
        replace (bool): Whether to sample with replacement
        
        Returns:
        numpy.array: The selected sample
        """
        return np.random.choice(self.population, size=n, replace=replace)

    def estimate_population_mean(self, sample):
        """
        Estimate the population mean from a sample.
        
        Args:
        sample (numpy.array): The sample
        
        Returns:
        float: Estimated population mean
        """
        return np.mean(sample)

    def estimate_population_total(self, sample):
        """
        Estimate the population total from a sample.
        
        Args:
        sample (numpy.array): The sample
        
        Returns:
        float: Estimated population total
        """
        return self.N * np.mean(sample)

    def estimate_variance(self, sample):
        """
        Estimate the variance of the sample mean.
        
        Args:
        sample (numpy.array): The sample
        
        Returns:
        float: Estimated variance of the sample mean
        """
        n = len(sample)
        s2 = np.var(sample, ddof=1)  # Sample variance
        return (self.N - n) / (self.N * n) * s2

    def confidence_interval(self, sample, confidence=0.95):
        """
        Calculate the confidence interval for the population mean.
        
        Args:
        sample (numpy.array): The sample
        confidence (float): Confidence level
        
        Returns:
        tuple: Lower and upper bounds of the confidence interval
        """
        n = len(sample)
        mean = np.mean(sample)
        se = np.sqrt(self.estimate_variance(sample))
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * se
        return mean - margin, mean + margin

    def visualize_sampling_distribution(self, sample_size, num_samples=1000):
        """
        Visualize the sampling distribution of the sample mean.
        
        Args:
        sample_size (int): Size of each sample
        num_samples (int): Number of samples to generate
        """
        means = [np.mean(self.select_sample(sample_size)) for _ in range(num_samples)]
        plt.figure(figsize=(10, 6))
        plt.hist(means, bins=30, edgecolor='black')
        plt.title(f"Sampling Distribution of Mean (n={sample_size})")
        plt.xlabel("Sample Mean")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(self.population), color='red', linestyle='dashed', linewidth=2, label='Population Mean')
        plt.legend()
        plt.show()

    def compare_with_population(self, sample):
        """
        Compare the sample distribution with the population distribution.
        
        Args:
        sample (numpy.array): The sample
        """
        plt.figure(figsize=(12, 6))
        plt.hist(self.population, bins=50, alpha=0.5, label='Population', density=True)
        plt.hist(sample, bins=30, alpha=0.5, label='Sample', density=True)
        plt.title("Comparison of Sample and Population Distributions")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a population using PopulationGenerator
    pop_gen = PopulationGenerator(size=100000)
    pop_gen.generate_normal(mean=100, std=20)

    # Create SimpleRandomSampling object
    srs = SimpleRandomSampling(pop_gen.population)

    # Select a sample
    sample_size = 1000
    sample = srs.select_sample(sample_size)

    # Estimate population mean and total
    est_mean = srs.estimate_population_mean(sample)
    est_total = srs.estimate_population_total(sample)

    print(f"True population mean: {np.mean(srs.population):.2f}")
    print(f"Estimated population mean: {est_mean:.2f}")
    print(f"Estimated population total: {est_total:.2f}")

    # Calculate confidence interval
    ci_lower, ci_upper = srs.confidence_interval(sample)
    print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")

    # Visualize sampling distribution
    srs.visualize_sampling_distribution(sample_size, num_samples=1000)

    # Compare sample with population
    srs.compare_with_population(sample)