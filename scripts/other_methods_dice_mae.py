import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage import zoom
from skimage import io
from skimage.util import img_as_float
from PIL import Image

def binary_loader(path):
    assert os.path.exists(path), f"`{path}` does not exist."
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)
    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt


def resize_image(img, new_shape):
    factors = [float(new_dim) / old_dim for new_dim, old_dim in zip(new_shape, img.shape)]
    resized_img = zoom(img, zoom=factors, order=1)
    return resized_img

def dice_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

def calculate_f_measure(prediction, ground_truth, beta=1):
    prediction = prediction.astype(np.bool)
    ground_truth = ground_truth.astype(np.bool)

    tp = np.sum(prediction & ground_truth)
    fp = np.sum(prediction & ~ground_truth)
    fn = np.sum(~prediction & ground_truth)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)
    f1_measure = (beta ** 2) * (precision * recall) / (precision + recall + 1e-8)

    return f1_measure

def mean_absolute_error(gt, pred):
    mae = np.mean(np.abs(pred - gt))
    return mae


def main(ground_truth_folder, prediction_folder, results_file):
    total_dice = 0
    total_f1 = 0
    total_mae = 0
    num_images = 0

    for file_name in os.listdir(ground_truth_folder):
        ground_truth_path = os.path.join(ground_truth_folder, file_name)
        #VT1000 GT .jpg not png
        # file_name = file_name.split(".")[0]+".png"
        file_name = file_name.split(".")[0] + '.jpg'
        prediction_path = os.path.join(prediction_folder, file_name)

        if os.path.exists(prediction_path):
            ground_truth = binary_loader(ground_truth_path)
            prediction = binary_loader(prediction_path)
            #
            # ground_truth[ground_truth < 0.5] = 0
            # ground_truth[ground_truth > 0.5] = 1
            if file_name == 'sun_bpuqnuvbhgkdttao.png':
                print(1)
            if prediction.size != ground_truth.size:
                prediction = prediction.resize(ground_truth.size, Image.BILINEAR)
            prediction, ground_truth = normalize_pil(pre=prediction, gt=ground_truth)

            dice = dice_coefficient(ground_truth, prediction)
            f1_measure = calculate_f_measure(ground_truth, prediction)

            mae = mean_absolute_error(ground_truth, prediction)
            total_dice += dice
            total_f1 += f1_measure
            total_mae += mae

            num_images += 1

            print(f"Processed: {file_name}")
            print(mae)

    average_dice = total_dice / num_images
    average_f1 = total_mae / num_images
    average_mae = total_mae / num_images

    mode = "a" if os.path.exists(results_file) else "w"
    with open(results_file, mode) as f:
        if mode == "w":
            f.write("Ground Truth Path\tPrediction Path\tNumber of Images\tAverage Dice Coefficient\tAverage F1\tAverage Mean Absolute Error\n")
        f.write(f"{ground_truth_folder}\t{prediction_folder}\t{num_images}\t{average_dice:.4f}\t{average_f1:.4f}\t{average_mae:.4f}\n")
        print(f"{ground_truth_folder}\t{prediction_folder}\t{num_images}\t{average_dice:.4f}\t{average_f1:.4f}\t{average_mae:.4f}\n")

if __name__ == "__main__":
    ground_truth_folder = "/home/david/dataset/ShadowDetect/SBU-Test_Dataset/ShadowMasks"
    prediction_folder = "/home/david/dataset/ShadowDetect/DSC-SBU"
    results_file = "eval_methods_results.txt"
    main(ground_truth_folder, prediction_folder, results_file)