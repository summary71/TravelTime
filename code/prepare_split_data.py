# prepare_split_data.py
import os, glob, json, argparse
from vrp_utils import load_data, split_data

# =============================
# 1. Argparse
# =============================
parser = argparse.ArgumentParser(description="Prepare and split VRP data, save to JSON.")
parser.add_argument("-dd", "--datadir", type=str, default="VRPdataSmall", help="Data directory containing raw VRP json files.")
parser.add_argument("-nd", "--numdata", type=int, default=-1, help="Number of data files to use (-1 for all).")
parser.add_argument("-in", "--inputnorm", type=int, default=0, help="Input normalization (0: no, 1: yes).")

args = parser.parse_args()


# =============================
# 2. Settings
# =============================
mydrive = "./"
data_directory = args.datadir
num_data = args.numdata

data_prefix = f"{data_directory}_in{args.inputnorm}"

# =============================
# 3. Load and split data
# =============================
vrpdatapath = mydrive + "../" + data_directory
json_files = sorted(glob.glob(vrpdatapath + "/*.json"), key=os.path.getctime, reverse=True)
recent_json_files = json_files[0:num_data] if num_data > 0 else json_files
print(f"Num Files: {len(recent_json_files)}")

# Load raw data
alldata = load_data(recent_json_files, args.inputnorm)

# Split
traindata, valdata, testdata = split_data(alldata)

# =============================
# 4. Save split data to JSON
# =============================
os.makedirs(f"{mydrive}split_data_{data_directory}", exist_ok=True)

with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_traindata.json", 'w') as f:
    json.dump(traindata.tolist(), f)
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_valdata.json", 'w') as f:
    json.dump(valdata.tolist(), f)
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_testdata.json", 'w') as f:
    json.dump(testdata.tolist(), f)

print(f"Split data saved to split_data_{data_directory}/ folder.")
