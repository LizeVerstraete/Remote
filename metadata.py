import os
import pickle

# Function to generate metadata from directory
def generate_metadata_from_directory(directory):
    metadata = []
    for dataset_folder in os.listdir(directory):
        dataset_path = os.path.join(directory, dataset_folder)
        if os.path.isdir(dataset_path):
            for patientset_folder in os.listdir(dataset_path):
                patientset_path = os.path.join(dataset_path, patientset_folder)
                if os.path.isdir(patientset_path):
                    for image_filename in os.listdir(patientset_path):
                        if image_filename.endswith(".jpg") or image_filename.endswith(".jpeg") or image_filename.endswith(".png"):
                            image_path = os.path.join(patientset_path, image_filename)
                            metadata.append({"path": image_path, "dataset": dataset_folder, "patientset": patientset_folder})
    return metadata

# Define the directory containing your data
data_directory = "/esat/biomeddata/kkontras/r0786880/data"

# Ensure the metadata directory exists
metadata_directory = "./"
os.makedirs(metadata_directory, exist_ok=True)

# Generate metadata for the entire dataset
metadata = generate_metadata_from_directory(data_directory)
unique_patientsets = sorted(set(item["patientset"] for item in metadata))

val_patients = unique_patientsets[:2]+unique_patientsets[-2:]
test_patients = unique_patientsets[2:4]+unique_patientsets[-4:-2]
train_patients = list(set(unique_patientsets)-set(val_patients)-set(test_patients))
# Split the metadata into train, val, and test sets
train_metadata = []
test_metadata = []
val_metadata = []
# Iterate over the metadata and filter out items for train_patients
for item in metadata:
    if item["patientset"] in test_patients:
        test_metadata.append(item)
    elif item["patientset"] in val_patients:
        val_metadata.append(item)
    else:
        train_metadata.append(item)

# Save the metadata for each split to a pickle file
with open(os.path.join(metadata_directory, '../Project_Template/datasets/Kinetics/metadata_train.pkl'), 'wb') as f:
    pickle.dump(train_metadata, f)
with open(os.path.join(metadata_directory, '../Project_Template/datasets/Kinetics/metadata_val.pkl'), 'wb') as f:
    pickle.dump(val_metadata, f)
with open(os.path.join(metadata_directory, '../Project_Template/datasets/Kinetics/metadata_test.pkl'), 'wb') as f:
    pickle.dump(test_metadata, f)