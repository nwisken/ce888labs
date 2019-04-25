import pandas as pd
import gc

# filepaths of dataset
CNAE_9_PATH = "CNAE_9.data"
HAR_DATA_PATH = "HAR_data.txt"
HAR_LABEL_PATH = "HAR_label.txt"
DIGIT_PATH = "digit_data.csv"

cnae= pd.read_csv(CNAE_9_PATH, header=None)

print("CNAE-9 characteristics ")
cnae_data= cnae.drop(cnae.columns[0], axis=1)
cnae_label = cnae[cnae.columns[0]]
cnae_describe = cnae_data.stack().describe()
cnae_median = cnae_data.stack().median()
cnae_mode = cnae_data.stack().mode()
cnae_std = cnae_data.stack().std()
print(cnae_describe)
print("mode", cnae_mode)
del cnae
del cnae_data

print("\nHuman activity characteristics ")
har_data = pd.read_csv(HAR_DATA_PATH, delim_whitespace=True)
har_labels = pd.read_csv(HAR_LABEL_PATH, delim_whitespace=True)
har_describe = har_data.stack().describe()
har_median = har_data.stack().median()
har_mode = har_data.stack().mode()
har_std = har_data.stack().std()
print(har_describe)
print("mode", har_mode)
del har_data
del har_labels

gc.collect()
digit= pd.read_csv(DIGIT_PATH)

print("\ndigit characteristics ")
digit_data= digit.drop(digit.columns[0], axis=1)
digit_label = digit[digit.columns[0]]
digit_describe = digit_data.stack().describe()
digit_median = digit_data.stack().median()
digit_mode = digit_data.stack().mode()
digit_std = digit_data.stack().std()
print(digit_describe)
print("mode", digit_mode)