import matplotlib

matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import bootstrap

VEHICLE_FILE = "vehicles.csv"

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
    df = pd.read_csv(VEHICLE_FILE)
    df = df[pd.notnull(df["New Fleet"])]

    print((df.columns))
    vehicle_scatter = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

    vehicle_scatter.axes[0, 0].set_ylim(0, )
    vehicle_scatter.axes[0, 0].set_xlim(0, )

    vehicle_scatter.savefig("vehiclescatter.png", bbox_inches='tight')
    vehicle_scatter.savefig("vehiclescatter.pdf", bbox_inches='tight')

    data_new_fleet = df.values.T[1]
    data_current_fleet = df.values.T[0]

    print((("Mean: %f") % (np.mean(data_new_fleet))))
    print((("Median: %f") % (np.median(data_new_fleet))))
    print((("Var: %f") % (np.var(data_new_fleet))))
    print((("std: %f") % (np.std(data_new_fleet))))
    print((("MAD: %f") % (mad(data_new_fleet))))

    plt.clf()
    vehicle_histogram = sns.distplot(data_new_fleet, bins=20, kde=False, rug=True).get_figure()

    axes = plt.gca()
    axes.set_xlabel('New Fleet Vehicle MPG (Miles Per Gallon)')
    axes.set_ylabel('Frequency')

    vehicle_histogram.savefig("vehiclehistogram.png", bbox_inches='tight')
    vehicle_histogram.savefig("vehiclehistogram.pdf", bbox_inches='tight')

    mean_current_fleet = np.mean(data_current_fleet)
    mean_new_fleet = np.mean(data_new_fleet)
    boot_current = bootstrap.boostrap(data_current_fleet,data_current_fleet.shape[0],1000)
    boot_new = bootstrap.boostrap(data_new_fleet, data_current_fleet.shape[0], 1000)

    print("Current Fleet: Mean:{} Lower: {} Upper: {}".format(mean_current_fleet, boot_current[1], boot_current[2]))
    print("New Fleet: Mean:{} Lower: {} Upper: {}".format(mean_new_fleet, boot_new[1], boot_new[2]))
