# Adapted from https://github.com/imigraz/GAFFA

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import os
import numpy as np

plt.rcParams["savefig.bbox"] = 'tight'

little_finger = [13, 33, 34, 35, 36]
ring_finger = [14, 29, 30, 31, 32]
middle_finger = [15, 25, 26, 27, 28]
index_finger = [16, 21, 22, 23, 24]
thumb_finger = [17, 18, 19, 20, -1]

all_fingers = torch.tensor([little_finger, ring_finger, middle_finger, index_finger, thumb_finger])


def load_csv_landmarks(file_name):
    import csv
    num_landmarks = 37
    landmarks_dict = {}
    dim = 2
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(
                row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries,
                                                                                                     len(row))
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    pass
                else:
                    if dim == 2:
                        coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                    elif dim == 3:
                        coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    landmarks.append(coords)
            landmarks_dict[id] = np.array(landmarks)

    return landmarks_dict


def get_neighbors(lst):
    if len(lst) < 2:
        return torch.tensor([])

    neighbors = [[lst[i], lst[i + 1]] for i in range(len(lst) - 1)]
    return torch.tensor(neighbors)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def draw_circle(image_tensor, center_x, center_y, radius):
    """
    Draw a circle on the input image.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor.
    - center_x (int): x-coordinate of the circle's center.
    - center_y (int): y-coordinate of the circle's center.
    - radius (int): Radius of the circle.
    - color (tuple): RGB color tuple for the circle (default is red).

    Returns:
    - torch.Tensor: Image tensor with the circle drawn.
    """
    image_tensor = torch.squeeze(image_tensor)

    # Generate coordinates for the circle
    aranges = [torch.arange(s, device='cuda') for s in image_tensor.shape]
    x, y = torch.meshgrid(*aranges, indexing='ij')

    # Use the circle equation to set pixels inside the circle to 1
    circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2

    noisy_image = torch.zeros_like(image_tensor)

    # Fill the circle in the original image
    image_tensor[circle_mask] = noisy_image[circle_mask]

    return torch.unsqueeze(image_tensor, 0), circle_mask


def simulate_occlusion_fingers_orig(image_tensor, landmark_locations, radius=10):
    for img_id in range(0, image_tensor.shape[0]):
        rand_fingers = torch.randint(low=0, high=len(all_fingers), size=(len(all_fingers),))
        rand_fingers = list(all_fingers[rand_fingers == 0])
        for i in range(0, len(rand_fingers)):
            rand_finger = rand_fingers[i]
            rand_finger = get_neighbors(rand_finger)
            rand_parts = torch.randint(low=0, high=len(rand_finger)-1, size=(4,))
            rand_fingers[i] = rand_finger[rand_parts == 0]
        for finger in rand_fingers:
            if len(finger) == 0:
                continue
            for first_landmark_id, second_landmark_id in finger:
                if -1 in second_landmark_id:
                    continue
                image = image_tensor[img_id]
                # get center coordinates of neighbouring finger landmarks
                first_x = landmark_locations[img_id][first_landmark_id][1]
                first_y = landmark_locations[img_id][first_landmark_id][2]
                second_x = landmark_locations[img_id][second_landmark_id][1]
                second_y = landmark_locations[img_id][second_landmark_id][2]

                vector = torch.tensor([first_x - second_x, first_y - second_y], device='cuda')
                distance = torch.norm(vector)
                amount_circles = ((distance / radius) + 1).to(dtype=torch.int)

                part_way = vector * (1.0 / amount_circles)
                draw_circle(image, first_x, first_y, radius)
                draw_circle(image, second_x, second_y, radius)
                for i in range(0, amount_circles):
                    image = draw_circle(image, second_x + part_way[0] * i, second_y + part_way[1] * i, radius)
                image_tensor[img_id] = image
                save_image(image_tensor[img_id], str(img_id) + ".png")

    return image_tensor


def simulate_occlusion_fingers(image_tensor, landmark_locations, img_id=0, output_dir=None, radius=10):
    # image tensor shape: (1, W, H)
    rand_fingers = torch.randint(low=0, high=len(all_fingers), size=(len(all_fingers),))
    rand_fingers = list(all_fingers[rand_fingers == 0])

    for i in range(0, len(rand_fingers)):
        rand_finger = rand_fingers[i]
        rand_finger = get_neighbors(rand_finger)
        rand_parts = torch.randint(low=0, high=len(rand_finger)-1, size=(4,))
        rand_fingers[i] = rand_finger[rand_parts == 0]
        
    circle_masks = []
    degradation_mask = torch.zeros(image_tensor.shape)
    for finger in rand_fingers:
        if len(finger) == 0:
            continue
        for first_landmark_id, second_landmark_id in finger:
            if -1 in second_landmark_id:
                continue
            image = image_tensor
            # get center coordinates of neighbouring finger landmarks
            first_x = landmark_locations[first_landmark_id][1]
            first_y = landmark_locations[first_landmark_id][0]
            second_x = landmark_locations[second_landmark_id][1]
            second_y = landmark_locations[second_landmark_id][0]

            vector = torch.tensor([first_x - second_x, first_y - second_y], device='cuda')
            distance = torch.norm(vector)
            amount_circles = ((distance / radius) + 1).to(dtype=torch.int)

            part_way = vector * (1.0 / amount_circles)
            _, circle_mask = draw_circle(image, first_x, first_y, radius)
            circle_masks.append(circle_mask)
            _, circle_mask = draw_circle(image, second_x, second_y, radius)
            circle_masks.append(circle_mask)

            for i in range(0, amount_circles):
                image, circle_mask = draw_circle(image, second_x + part_way[0] * i, second_y + part_way[1] * i, radius)
                circle_masks.append(circle_mask)
            image_tensor = image

            # degradation mask for inpainting
            degradation_mask = 1 - torch.any(torch.stack(circle_masks), dim=0).to(torch.int).unsqueeze(0)

            if output_dir is not None:
                print("save image....")
                save_image(image_tensor, os.path.join(output_dir, str(img_id) + ".png"))

    return image_tensor, degradation_mask.unsqueeze(0)


def simulate(x, landmarks):
    occluded_image, mask = simulate_occlusion_fingers(x[:, 0, :, :], landmarks)

    return occluded_image.to("cuda"), mask.to("cuda")
