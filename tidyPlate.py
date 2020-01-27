# In[1]:
import numpy as np
import pandas as pd

excel_file = "./sampleData.xlsx"
layout_df = pd.read_excel(excel_file, sheet_name="Input", index_col=0, nrows=8)
layout_df.index.names = [None]
raw_df = pd.read_excel(
    excel_file, sheet_name="Input", index_col=0, skiprows=11, nrows=8
)
raw_df.index.names = [None]

combined_df = pd.DataFrame(
    {
        "row": pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"]).repeat(12),
        "col": pd.np.tile(np.array(np.arange(1, 13)), 8),
    },
    columns=["row", "col"],
)

combined_df["well"] = combined_df["row"] + combined_df["col"].map(str)
combined_df.reset_index(drop=True, inplace=True)

labels_ser = [""] * 96
raw_ser = [0] * 96
for index, row in combined_df.iterrows():
    labels_ser[index] = layout_df[row["col"]][row["row"]]
    raw_ser[index] = raw_df[row["col"]][row["row"]]
combined_df["label"] = labels_ser
combined_df["raw"] = raw_ser
# Combined dataframe containing each well along with its corresponding label and raw data

STD_BLANK = combined_df.loc[combined_df["label"] == "STD BLANK"]["raw"].mean()
SAMP_BLANK = combined_df.loc[combined_df["label"] == "BLANK"]["raw"].mean()

minBlk_ser = [0] * 96
for index, row in combined_df.iterrows():
    if "STD" in row["label"]:
        if STD_BLANK != 0:
            minBlk_ser[index] = row["raw"] - STD_BLANK
        else:
            minBlk_ser[index] = row["raw"] - SAMP_BLANK
    else:
        minBlk_ser[index] = row["raw"] - SAMP_BLANK
combined_df["minusBlank"] = minBlk_ser
# Added 'minusBlank' values to combined_df dataframe

# Read in user-input preferences for Std Curve from the xlsx
user_std_options = pd.read_excel(
    excel_file, sheet_name="Input", index_col=0, skiprows=22, nrows=4, usecols=[0, 1]
)
user_std_options.index.names = [None]

# Generate a dataframe to contain Std Curve information
std_df = pd.DataFrame()
if user_std_options["userPref"]["Serial Diluted (y/n)?"] == "y":
    std_df = pd.read_excel(
        excel_file,
        sheet_name="Input",
        index_col=1,
        skiprows=28,
        nrows=user_std_options["userPref"]["# of Concentrations:"],
        usecols=[0, 1, 2],
    )
else:
    std_df = pd.read_excel(
        excel_file,
        sheet_name="Input",
        index_col=1,
        skiprows=46,
        nrows=user_std_options["userPref"]["# of Concentrations:"],
        usecols=[0, 1, 2],
    )

std_rep_dict = {}
for name, row in std_df.iterrows():
    if name not in std_rep_dict.keys():
        std_rep_dict[name] = []

for index, row in combined_df.iterrows():
    if row["label"] in std_rep_dict.keys():
        std_rep_dict[row["label"]].append(row["minusBlank"])

std_rep_df = pd.DataFrame.from_dict(std_rep_dict, orient="index")

STD_REPS = 0
for key in std_rep_dict.keys():
    if len(std_rep_dict[key]) > STD_REPS:
        STD_REPS = len(std_rep_dict[key])

col_rep_names = []
for c in range(STD_REPS):
    col_rep_names.append("Rep" + str(c + 1))
std_rep_df.columns = col_rep_names

std_df = pd.concat([std_df, std_rep_df], axis=1)

# Fetch list of col names representing Std Curve replicates
std_df_rep_cols = [col for col in std_df.columns if "Rep" in col]


# Generate columns for mean, std deviation, and coefficient of variation
std_df["avg"] = std_df.loc[:, std_df_rep_cols].mean(axis=1)
std_df["sd"] = std_df.loc[:, std_df_rep_cols].std(axis=1)
std_df["cv"] = std_df.loc[:, "sd"] / std_df.loc[:, "avg"]
print("std_df")
std_df


# In[2]:


import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import leastsq


def logistic5(x, A, B, C, D, E):
    """5PL logistic equation"""
    return D + ((A - D) / (np.float_power((1 + (np.float_power((x / C), B))), E)))


def residuals(p, y, x):
    """Deviations of data from fitted 5PL Curve"""
    A, B, C, D, E = p  # p is array values for each coefficient
    err = y - logistic5(x, A, B, C, D, E)  # Array of residuals: measured y - calculated
    return err


def peval(x, p):
    """Evaluated value at x with current parameters"""
    A, B, C, D, E = p
    return logistic5(x, A, B, C, D, E)


x = []
y_meas = []
for ind, row in std_df.iterrows():
    for rep in std_df_rep_cols:
        if not pd.isnull(row[rep]):
            x.append(row["conc"])
            y_meas.append(row[rep])

x = np.asarray(x)
y_meas = np.asarray(y_meas)

print("x")
print(x)
print("y")
print(y_meas)


# A, B, C, D, E = 0.5, 2.5, 8, 7.3, 1
A = np.amax([np.amin(minBlk_ser), 0.0001])  # Min asymptote
D = np.amax(y_meas)  # Max asymptote
B = (D - A) / np.amax(x)  # Steepness
C = np.amax(x) / 2  # Inflection Point, conc at which y = (D - A) / 2
E = 1  # Asymmetry factor
print(A, B, C, D, E)


# In[3]:


# Initial guess for parameters
p0 = [A, B, C, D, E]


# Fit equation using least squares optimization
plsq = leastsq(func=residuals, x0=p0, args=(y_meas, x))


# In[4]:

y_true = logistic5(x, A, B, C, D, E)

# Plot results
plt.plot(x, peval(x, plsq[0]), x, y_meas, "o", x, y_true)
plt.title("Least-squares 5PL fit to Std Curve")
plt.legend(["Fit", "Measured", "True"], loc="upper left")
for i, (param, actual, est) in enumerate(zip("ABCDE", [A, B, C, D, E], plsq[0])):
    plt.text(10, 3 - i * 0.5, "%s = %.2f, est(%s) = %.2f" % (param, actual, param, est))
# plt.savefig('logistic.png')


# %%
