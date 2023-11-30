import os

for folder in ["train", "test", "val"]:
    for file in os.listdir(os.path.join("FaceDetection", "data", folder, "images")):
        filename = file.split(".")[0] + ".json"
        existing_filepath = os.path.join("FaceDetection", "data", "labels", filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join(
                "FaceDetection", "data", folder, "labels", filename
            )
            os.replace(existing_filepath, new_filepath)