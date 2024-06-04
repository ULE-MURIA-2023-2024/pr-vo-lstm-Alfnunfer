
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(
                    rgb_paths, ground_truth_data)

            # TODO: create sequences
            for i in range(len(rgb_paths) - sequence_length + 1):
                if not validation:
                    sequence = {
                        "images": rgb_paths[i:i + sequence_length],
                        "ground_truth": interpolated_ground_truth[i:i + sequence_length]
                    }
                else:
                    sequence = {
                        "images": rgb_paths[i:i + sequence_length],
                        "ground_truth": []
                    }
                self.sequences.append(sequence)

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_data = self.sequences[idx]
        sequence_images = []
        ground_truth_pos = []
        timestamp = 0

        # TODO: return the next sequence
        for img_path in sequence_data["images"]:
            timestamp, path = img_path
            img = cv2.imread(path)
            if self.transform:
                img = self.transform(img)
            sequence_images.append(img)
        
        

        if not self.validation:
            ground_truth_pos = [gt[1] for gt in sequence_data["ground_truth"]]

            ground_truth_pos = [
                ground_truth_pos[-1][0] - ground_truth_pos[0][0],
                ground_truth_pos[-1][1] - ground_truth_pos[0][1],
                ground_truth_pos[-1][2] - ground_truth_pos[0][2],
                ground_truth_pos[-1][3] - ground_truth_pos[0][3],
                ground_truth_pos[-1][4] - ground_truth_pos[0][4],
                ground_truth_pos[-1][5] - ground_truth_pos[0][5],
                ground_truth_pos[-1][6] - ground_truth_pos[0][6]
                ]


        sequence_images = torch.stack(sequence_images)
        
        
        
        
      
            
        new_ground_truth_pos = torch.tensor(ground_truth_pos, dtype=torch.float32)

        
        
        return sequence_images, new_ground_truth_pos, timestamp

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
