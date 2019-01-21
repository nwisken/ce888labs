import matplotlib

matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations):
    # <---INSERT YOUR CODE HERE--->
    new_samples_index = np.random.randint(0,sample_size, (iterations,sample_size))
    iteration_means = np.zeros(iterations)
    new_samples = sample[new_samples_index]
    data_mean = np.mean(new_samples)

    for iteration in range(iterations):
        iteration_data = new_samples[iteration, :]
        iteration_mean = np.mean(iteration_data)
        iteration_means[iteration] = iteration_mean

    lower = np.percentile(iteration_means,5)
    upper = np.percentile(iteration_means, 95)
    return data_mean, lower, upper


if __name__ == "__main__":
    df = pd.read_csv('./salaries.csv')

    data = df.values.T[1]
    boots = []
    for i in range(100, 100000, 1000):
        boot = boostrap(data, data.shape[0], i)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
    sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
    sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')

# print ("Mean: %f")%(np.mean(data))
# print ("Var: %f")%(np.var(data))
