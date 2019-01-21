import numpy as np

def power(sample1, sample2, reps, size, alpha):
    times_greater = 0
    mean_sample_1 = np.mean(sample1)
    mean_sample_2 = np.mean(sample2)
    obsv = mean_sample_2 - mean_sample_1

    new_samples_index1 = np.random.randint(0, size, (reps, size))
    new_samples_index2 = np.random.randint(0, size, (reps, size))
    new_samples1 = sample1[new_samples_index1]
    new_samples2 = sample2[new_samples_index2]

    for rep in range(reps):
        s1 = new_samples1[rep, :]
        s2 = new_samples2[rep, :]
        s1_mean = np.mean(s1)
        s2_mean = np.mean(s2)
        perm = s2_mean-s1_mean
        if perm > obsv:
            times_greater+=1


    p = times_greater/reps
    