import cv2
import numpy as np

def best_fit_img(input_path, candidate_paths):
    # Load input image
    input_img = cv2.imread(input_path)

    # Convert to LAB color space
    lab_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

    # Calculate color histogram of input image
    hist = cv2.calcHist([lab_img], [1, 2], None, [128, 128], [0, 128, 0, 128])

    # Load a set of candidate images
    candidate_imgs = candidate_paths

    # Calculate color histogram of each candidate image
    candidate_hists = []
    for candidate_img in candidate_imgs:
        candidate_img = cv2.imread(candidate_img)
        candidate_lab_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2LAB)
        candidate_hist = cv2.calcHist([candidate_lab_img], [1, 2], None, [128, 128], [0, 128, 0, 128])
        candidate_hists.append(candidate_hist)

    # Compare histograms using correlation distance 
    # (which should be higher for better correlation, so therefore the minus)
    distances = []
    for candidate_hist in candidate_hists:
        distance = cv2.compareHist(hist, candidate_hist, cv2.HISTCMP_CORREL)
        distances.append(-distance)

    # Sort candidate images based on distance
    sorted_candidates = [candidate_imgs[i] for i in np.argsort(distances)]

    # Select the most similar image
    most_similar_imgs = sorted_candidates

    # Return the best images in order
    return most_similar_imgs

if __name__ == "__main__":
    input_path = "porsche.png"
    candidate_paths = ["imag1.jpg", "imag2.jpg", "imag3.jpg"]
    print(best_fit_img(input_path, candidate_paths))