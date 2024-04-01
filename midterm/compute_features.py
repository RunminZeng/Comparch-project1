import cv2
import numpy as np
import multiprocessing
import random
from functools import partial
import os

def compute_sift_for_image(image_path):
    """
    Worker function to compute SIFT descriptors for a single image.
    """
    img = cv2.imread(image_path)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(img, None)
    return des
def compute_sift_features_parallel(save_dir, sample_size=1000):
    """
    Compute SIFT features for a random sample of images in the data directory using multiprocessing.
    """

    image_paths = [os.path.join(save_dir, f"{i}.jpg") for i in range(len(os.listdir(save_dir)))]
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Map the worker function over the image paths and collect all descriptors
    all_descriptors = pool.map(compute_sift_for_image, image_paths)
    
    # Close the pool and wait for work to finish
    pool.close()
    pool.join()
    
    # Flatten the list of descriptors
    sift_descriptors = [des for sublist in all_descriptors if sublist is not None for des in sublist]
    
    # Random sampling of descriptors to reduce the dataset size
    if len(sift_descriptors) > sample_size:
        sift_descriptors = random.sample(sift_descriptors, sample_size)
    
    return np.asarray(sift_descriptors)

def get_VLAD(img, codebook):
    """
    Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
    """
    sift = cv2.SIFT_create()
    
    _, des = sift.detectAndCompute(img, None)
    pred_labels = codebook.predict(des)
    centroids = codebook.cluster_centers_
    k = codebook.n_clusters
    VLAD_feature = np.zeros([k, des.shape[1]])
    for i in range(k):
        if np.sum(pred_labels == i) > 0:
            VLAD_feature[i] = np.sum(des[pred_labels == i, :] - centroids[i], axis=0)
    VLAD_feature = VLAD_feature.flatten()
    VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
    VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)
    return VLAD_feature

def get_VLAD_wrapper(img_path, codebook):
    """
    A wrapper function that loads the image, creates a SIFT object,
    and then computes the VLAD descriptor.
    """
    img = cv2.imread(img_path)
    return get_VLAD(img, codebook)

def compute_vlad_descriptors_parallel(save_dir, count, codebook):
    """
    Use multiprocessing to compute VLAD descriptors for all images in parallel.
    Note: SIFT object is created inside each worker process.
    """
    # Create a list of image paths
    image_paths = [os.path.join(save_dir, f"{i}.jpg") for i in range(count)]
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Use partial to create a version of the wrapper function with codebook already set
    worker_function = partial(get_VLAD_wrapper, codebook=codebook)

    # Map the worker function over the image paths
    database = pool.map(worker_function, image_paths)
    
    # Close the pool and wait for work to finish
    pool.close()
    pool.join()
    
    return database