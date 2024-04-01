# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
from compute_features import *

def get_neighbor(img, tree, codebook):
    """
    Find the nearest neighbor in the database based on VLAD descriptor and refine with RANSAC.
    """
    sift = cv2.SIFT_create()
    
    q_VLAD = get_VLAD(img, codebook).reshape(1, -1)
    _, index = tree.query(q_VLAD, 1)
    nearest_id = index[0][0]
    # return nearest_id
    # Assuming you have a way to get the original keypoints and descriptors for the nearest image
    save_dir = "data/images/"
    nearest_img_path = os.path.join(save_dir, f"{nearest_id}.jpg")
    nearest_img = cv2.imread(nearest_img_path)

    # Compute keypoints and descriptors for the query image
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(nearest_img, None)  # Key points of the nearest image

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]

    # Minimum number of good matches to consider a reliable localization
    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Apply RANSAC
        _, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if inliers is not None and inliers.sum() > MIN_MATCH_COUNT:
            # print("RANSAC verification successful")
            return nearest_id
        # else:
            # print("RANSAC verification failed, looking for next best match")
            # Implement logic to handle failure (e.g., try the second nearest neighbor)
    # else:
        # print("Not enough good matches")

    return None


def show_target_images(images):
    """
    Display front, right, back, and left views of target location in 2x2 grid manner
    """
    if images is None or len(images) <= 0:
        return

    hor1 = cv2.hconcat(images[:2])
    hor2 = cv2.hconcat(images[2:])
    concat_img = cv2.vconcat([hor1, hor2])

    w, h = concat_img.shape[:2]

    color = (0, 0, 0)

    concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
    concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

    w_offset = 25
    h_offset = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    size = 0.75
    stroke = 1

    cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
    cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
    cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
    cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

    cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
    cv2.waitKey(1)

def display_img_from_id(id, window_name, save_dir):
    """
    Display image from database based on its ID using OpenCV
    """
    path = save_dir + str(id) + ".jpg"
    img = cv2.imread(path)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()

        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to

        # Load pre-trained codebook for VLAD encoding
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = []
        self.prenav = False
        self.goal = None
        self.index = 0
        self.frame_count = 0
        
        self.exploration_start_time = None
        self.navigation_start_time = None
        
        save_dir_full = os.path.join(os.getcwd(), self.save_dir)
        self.save_dir = save_dir_full
        if not os.path.isdir(save_dir_full):
            os.makedirs(save_dir_full, exist_ok=True)
        else:
            # clear images
            for file in os.listdir(save_dir_full):
                os.remove(os.path.join(save_dir_full, file))

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    show_target_images(self.get_target_images())
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        show_target_images(images)

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID.
        Now uses MiniBatchKMeans for faster codebook generation.
        """
        if self.count > 0:
            print("Compute sift features")
            sift_descriptors = compute_sift_features_parallel(self.save_dir, sample_size=10000)  # Adjust sample size as needed
            # Use MiniBatchKMeans for faster clustering
            print("Compute Kmeans")
            codebook = MiniBatchKMeans(n_clusters=64, batch_size=1000, n_init=10, verbose=1).fit(sift_descriptors)
            
            print("Dump codebook")
            pickle.dump(codebook, open("codebook.pkl", "wb"))
            self.codebook = codebook  # Update the codebook with the newly generated one
            # Update the database with VLAD descriptors for all images
            print("Compute VLAD")
            self.database = compute_vlad_descriptors_parallel(self.save_dir, self.count, self.codebook)
            
            # Create the BallTree
            print("Build BallTree")
            tree = BallTree(self.database, leaf_size=40)  # Adjusted leaf_size for potentially better performance
            self.tree = tree
            
            self.prenav = True
            
            print("Pre-navigation computations done.")

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        if not self.prenav:
            self.pre_nav_compute()

    def see(self, fpv):
        """
        Set the first-person view input
        """
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
        
        scale = 4

        h, w, _ = fpv.shape
        if self.screen is None:
            pygame.font.init()  # Ensure the font module is initialized
            self.screen = pygame.display.set_mode((w * scale, h * scale))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert and scale OpenCV images for Pygame.
            """
            # Scale image for 2x larger display
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        
        # Convert and blit the FPV image first
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(pygame.transform.scale(rgb, (w * scale, h * scale)), (0, 0))

        self.frame_count += 1
        
        current_time = pygame.time.get_ticks()
        phase_time = 0  # Default phase time
        
        font = pygame.font.Font(None, 36)
        
        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                if self.exploration_start_time is None:
                    self.exploration_start_time = current_time
                else:
                    phase_time = (current_time - self.exploration_start_time) // 1000
                    
                keys = pygame.key.get_pressed()
                
                if keys[pygame.K_q]:
                    print("q")
                    self.pre_nav_compute()

                if not self.prenav:
                    # if self.frame_count % 1 == 0:
                    save_path = self.save_dir + str(self.count) + ".jpg"
                    cv2.imwrite(save_path, fpv)
                    VLAD = get_VLAD(self.fpv, self.codebook)
                    self.database.append(VLAD)
                    self.count += 1
                    
            elif self._state[1] == Phase.NAVIGATION:
                if self.navigation_start_time is None:
                    self.navigation_start_time = current_time
                    self.exploration_start_time = None  # Reset exploration start time for future phases if needed
                phase_time = (current_time - self.navigation_start_time) // 1000
                    
                if not self.goal:
                    # Find out ID of all 4 views of target
                    targets = self.get_target_images()
                    self.goal = [get_neighbor(targets[i], self.tree, self.codebook) for i in range(4)]
                    print(f'Goal ID: {self.goal}')
                
            if self.prenav: # Display the goal IDs if pre-navigation is done
                if self.frame_count % 5 == 0:
                    self.index = get_neighbor(self.fpv, self.tree, self.codebook)
                    
                if self.index is None:
                    save_path = self.save_dir + str(self.count) + ".jpg"
                    cv2.imwrite(save_path, fpv)
                    VLAD = get_VLAD(self.fpv, self.codebook)
                    self.database.append(VLAD)
                    self.count += 1
                
                if self.goal:
                    ids_text = f"Current ID: {self.index}, Goal IDs: {self.goal}"
                else:
                    ids_text = f"Current ID: {self.index}"
                    
                ids_surface = font.render(ids_text, True, (255, 0, 0))
                self.screen.blit(ids_surface, (10, 90))
                
        # Display the current phase, the time spent in the current phase, and IDs at the top of the game window
        phase_text = f"Phase: {self._state[1].name}" if self._state else "Phase: Unknown"
        time_text = f"Time in Phase: {phase_time}s"  # Display time spent in the current phase
        phase_surface = font.render(phase_text, True, (255, 0, 0))
        time_surface = font.render(time_text, True, (255, 0, 0))
        self.screen.blit(phase_surface, (10, 10))
        self.screen.blit(time_surface, (10, 50))

        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())