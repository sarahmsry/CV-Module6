"""
Complete Structure from Motion (SfM) Pipeline
Reconstructs 3D object structure from multiple uncalibrated images.

Pipeline Overview:
1. Load calibrated camera intrinsics (K matrix)
2. Load and undistort images
3. Extract SIFT keypoints from all images
4. Build feature correspondences between image pairs
5. Compute Fundamental and Essential matrices
6. Recover camera poses (rotation R, translation t)
7. Triangulate 3D points from image correspondences
8. Refine and filter 3D point cloud
9. Estimate object boundary via convex hull
10. Visualize 3D reconstruction with camera trajectory
"""

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import glob

# ============================================================================
# CAMERA CALIBRATION & IMAGE LOADING
# ============================================================================

def load_calibration(calib_file='calibration.pkl'):
    """Load camera intrinsics K and distortion coefficients from calibration."""
    with open(calib_file, 'rb') as f:
        calib_data = pickle.load(f)
    K = calib_data['K']
    dist = calib_data['dist']
    cal_size = calib_data['cal_size']  # Size at which calibration was computed
    return K, dist, cal_size


def load_images(image_paths, undistort=True):
    """
    Load images from file paths and optionally undistort using camera matrix.
    
    Args:
        image_paths: List of image file paths
        undistort: Whether to apply distortion correction
    
    Returns:
        images: List of undistorted RGB images
        gray_images: List of grayscale versions for feature detection
    """
    K, dist, cal_size = load_calibration()
    K = np.float32(K)  # Ensure K is float32
    images = []
    gray_images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not load {path}")
            continue
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Scale K matrix if image size differs from calibration size
        K_scaled = K.copy()
        if (w, h) != cal_size:
            scale_x = w / cal_size[0]
            scale_y = h / cal_size[1]
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
        
        # Undistort if requested
        if undistort:
            img = cv2.undistort(img, K_scaled, dist)
        
        # Convert to RGB and grayscale
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        images.append(img_rgb)
        gray_images.append(gray)
    
    print(f"Loaded {len(images)} images, shape: {images[0].shape}")
    return images, gray_images, np.float32(K_scaled)


# ============================================================================
# FEATURE DETECTION & MATCHING
# ============================================================================

def detect_sift_features(gray_img, n_features=5000):
    """
    Detect SIFT keypoints and compute descriptors.
    
    Args:
        gray_img: Grayscale image
        n_features: Maximum number of features to detect
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: SIFT descriptor vectors (Nx128)
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2, ratio_threshold=0.8, 
                   min_matches=8):
    """
    Match SIFT features between two images using FLANN KD-tree matching.
    
    Mathematical basis:
    - FLANN (Fast Library for Approximate Nearest Neighbors) finds the k-NN
      for each descriptor in image1 among all descriptors in image2
    - Lowe's ratio test: For each match, compare distance to nearest neighbor
      versus distance to second-nearest neighbor. Keep only matches where
      ratio = dist(1st)/(2nd) < threshold. This filters out ambiguous matches
      that could be false positives.
    
    Args:
        descriptors1, descriptors2: SIFT descriptors from two images
        ratio_threshold: Lowe's ratio for filtering (0.8 = 80% better)
        min_matches: Minimum matches required to proceed
    
    Returns:
        matches: List of good match pairs (cv2.DMatch objects)
    """
    # FLANN KD-tree matcher with multiple parallel searches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # k-NN matching: find 2 nearest neighbors for each descriptor
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lowe's ratio test: keep only confident matches
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    print(f"  Matched: {len(good_matches)} features (kept {len(good_matches)} after ratio test)")
    
    if len(good_matches) < min_matches:
        print(f"  WARNING: Only {len(good_matches)} matches, need {min_matches}")
        return []
    
    return good_matches


def extract_matched_points(kpts1, kpts2, matches):
    """
    Extract point coordinates from matched keypoints.
    
    Args:
        kpts1, kpts2: Keypoint lists from two images
        matches: Good match pairs
    
    Returns:
        pts1, pts2: Nx2 arrays of matched point coordinates (normalized to [0,1])
    """
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
    return pts1, pts2


# ============================================================================
# EPIPOLAR GEOMETRY & POSE RECOVERY
# ============================================================================

def compute_fundamental_matrix(pts1, pts2, method='8point'):
    """
    Compute Fundamental matrix F from point correspondences.
    
    Mathematical foundation (epipolar constraint):
    For a point p1 in image1 and corresponding point p2 in image2,
    the epipolar constraint is: p2^T F p1 = 0 (in homogeneous coordinates)
    
    This means: p2 must lie on the epipolar line defined by F and p1.
    
    The 8-point algorithm:
    - Normalizes points to [-1, 1] range for numerical stability
    - Sets up linear system Af = 0 where each correspondence adds one row
    - Solves via SVD: A = USV^T, solution is rightmost column of V
    - Enforces rank-2 constraint: set smallest singular value to 0
    - Denormalizes result
    
    Args:
        pts1, pts2: Matched point coordinates (Nx2)
        method: 'ransac' for outlier-robust, '8point' for least-squares
    
    Returns:
        F: 3x3 Fundamental matrix
        inlier_pts1, inlier_pts2: Point pairs that satisfy epipolar constraint
    """
    # Use RANSAC for robustness (outliers are inevitable in real images)
    F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, 
                                             method=cv2.FM_RANSAC,
                                             ransacReprojThreshold=4.0,
                                             confidence=0.99)
    
    if F is None:
        print("  ERROR: Could not compute Fundamental matrix")
        return None, pts1, pts2
    
    # Extract points that satisfy epipolar constraint
    inlier_pts1 = pts1[inlier_mask.ravel() == 1]
    inlier_pts2 = pts2[inlier_mask.ravel() == 1]
    
    # Verify epipolar constraint
    pts1_h = np.column_stack([inlier_pts1, np.ones(len(inlier_pts1))])
    pts2_h = np.column_stack([inlier_pts2, np.ones(len(inlier_pts2))])
    epipolar_error = np.mean(np.abs(np.diag(pts2_h @ F @ pts1_h.T)))
    print(f"  Epipolar constraint error: {epipolar_error:.6f} px (should be ~0)")
    
    return F, inlier_pts1, inlier_pts2


def compute_essential_matrix(F, K):
    """
    Compute Essential matrix E from Fundamental matrix and camera intrinsics.
    
    Mathematical relationship: E = K^T F K
    
    The Essential matrix encodes only the geometric relationship (R, t) between
    cameras, while the Fundamental matrix includes camera intrinsics.
    
    Properties of E:
    - Rank 2 (one singular value is 0)
    - Singular values are (1, 1, 0) after normalization
    - Satisfies: E = [t]_x R where [t]_x is the skew-symmetric matrix of t
    
    Args:
        F: 3x3 Fundamental matrix
        K: 3x3 Camera intrinsic matrix
    
    Returns:
        E: 3x3 Essential matrix
    """
    E = K.T @ F @ K
    
    # Enforce rank-2 constraint via SVD
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0  # Set smallest singular value to 0
    E = U @ np.diag(S) @ Vt
    
    return E


def recover_pose(E, pts1, pts2, K):
    """
    Recover camera pose (R, t) from Essential matrix.
    
    Mathematical principle (SVD decomposition):
    The Essential matrix E = [t]_x R has rank 2 with SVD: E = USV^T
    This encodes rotation and translation (up to scale).
    
    There are 4 possible (R, t) solutions from E. We pick the one where
    triangulated points have positive depth (Z > 0) in both cameras.
    This is the "cheirality constraint".
    
    Args:
        E: 3x3 Essential matrix
        pts1, pts2: Matched point coordinates (Nx2)
        K: 3x3 Camera intrinsic matrix
    
    Returns:
        R: 3x3 Rotation matrix (det(R)=1, R^T R = I)
        t: 3x1 Translation vector (unit norm, scale ambiguous)
        mask: Boolean array indicating points with positive depth
    """
    # OpenCV's recoverPose uses RANSAC internally and cheirality filtering
    success, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=None)
    
    if success < 8:  # recoverPose returns number of inliers
        print("  WARNING: Pose recovery found < 8 points with positive depth")
    
    # Verify R is a valid rotation matrix
    det_R = np.linalg.det(R)
    print(f"  Rotation matrix det(R) = {det_R:.4f} (should be 1.0)")
    if det_R < 0:
        R = -R
    
    return R, t, mask


# ============================================================================
# TRIANGULATION & 3D POINT RECONSTRUCTION
# ============================================================================

def manual_triangulate(P1, P2, pts1, pts2):
    """
    Manual DLT triangulation when OpenCV fails.
    Solves the linear system for each pair of points.
    """
    n_points = len(pts1)
    points4d = np.zeros((4, n_points), dtype=np.float32)
    
    for i in range(n_points):
        # Build the 4x4 system for this point pair
        p1 = np.append(pts1[i], 1.0)
        p2 = np.append(pts2[i], 1.0)
        
        # Skew-symmetric matrices for cross product
        p1_x = np.array([
            [0, -p1[2], p1[1]],
            [p1[2], 0, -p1[0]],
            [-p1[1], p1[0], 0]
        ])
        
        p2_x = np.array([
            [0, -p2[2], p2[1]],
            [p2[2], 0, -p2[0]],
            [-p2[1], p2[0], 0]
        ])
        
        # Build system: A X = 0
        A = np.vstack([p1_x @ P1, p2_x @ P2])
        
        # Solve via SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]  # Rightmost column of V
        
        points4d[:, i] = X
    
    return points4d


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from two camera views using linear algebra.
    
    Mathematical method (DLT - Direct Linear Transform):
    Given two projection matrices P1, P2 and image coordinates p1, p2:
    
    P1 = K1[I|0]      (camera 1 at origin, identity rotation)
    P2 = K2[R|t]      (camera 2 with pose R,t relative to camera 1)
    
    Image point p1 projects from 3D point X via: p1 = P1 X (homogeneous)
    Image point p2 projects from 3D point X via: p2 = P2 X (homogeneous)
    
    Solving the system (using SVD):
    [p1_x P1[0,:] - P1[2,:]]     [X]
    [p1_y P1[1,:] - P1[2,:]]  *  [Y] = 0
    [p2_x P2[0,:] - P2[2,:]]     [Z]
    [p2_y P2[1,:] - P2[2,:]]     [W]
    
    Solution is rightmost column of V from SVD, normalized by W component.
    
    Args:
        P1, P2: 3x4 projection matrices (float32)
        pts1, pts2: Nx2 image coordinates in both views (float32, pre-filtered)
    
    Returns:
        points3d: Nx3 array of triangulated 3D points (X,Y,Z)
    """
    # Ensure proper format: contiguous C-order float32 matrices
    pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float32) 
    P1 = np.ascontiguousarray(P1, dtype=np.float32)
    P2 = np.ascontiguousarray(P2, dtype=np.float32)
    
    print(f"      P1: dtype={P1.dtype}, shape={P1.shape}, C_CONT={P1.flags['C_CONTIGUOUS']}")
    print(f"      P2: dtype={P2.dtype}, shape={P2.shape}, C_CONT={P2.flags['C_CONTIGUOUS']}")
    print(f"      pts1: dtype={pts1.dtype}, shape={pts1.shape}, C_CONT={pts1.flags['C_CONTIGUOUS']}")  
    print(f"      pts2: dtype={pts2.dtype}, shape={pts2.shape}, C_CONT={pts2.flags['C_CONTIGUOUS']}")
    print(f"      pts1.T: dtype={pts1.T.dtype}, shape={pts1.T.shape}")
    print(f"      pts2.T: dtype={pts2.T.dtype}, shape={pts2.T.shape}")
    
    # OpenCV triangulation (normalized DLT method)
    try:
        points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    except Exception as e:
        print(f"      ERROR in triangulatePoints: {e}")
        print(f"      Trying manual solution instead...")
        # If OpenCV fails, use manual DLT
        points4d = manual_triangulate(P1, P2, pts1, pts2)
    
    
    # Convert from homogeneous (X,Y,Z,W) to Euclidean (X/W, Y/W, Z/W)
    points3d = (points4d[:3] / points4d[3]).T
    
    return points3d


def filter_points_by_depth(points3d, z_min=0.1, z_max=100):
    """
    Filter 3D points by depth constraints.
    
    Args:
        points3d: Nx3 array of 3D points
        z_min, z_max: Valid depth range
    
    Returns:
        filtered_points: Points within depth range
        mask: Boolean mask of valid points
    """
    mask = (points3d[:, 2] > z_min) & (points3d[:, 2] < z_max)
    filtered_points = points3d[mask]
    return filtered_points, mask


def filter_outliers_statistical(points3d, n_sigma=3.0):
    """
    Remove statistical outliers using depth-based filtering.
    
    Identifies points whose Z-depth is more than n_sigma standard deviations
    from the median. These are likely reconstruction artifacts or background.
    
    Args:
        points3d: Nx3 array of 3D points
        n_sigma: Number of standard deviations for threshold
    
    Returns:
        filtered_points: Outliers removed
        mask: Boolean mask of retained points
    """
    z_values = points3d[:, 2]
    median_z = np.median(z_values)
    std_z = np.std(z_values)
    
    threshold = median_z + n_sigma * std_z
    mask = z_values < threshold
    
    print(f"    Z depth range: {z_values.min():.3f} to {z_values.max():.3f}")
    print(f"    Median Z: {median_z:.3f}, Std: {std_z:.3f}")
    print(f"    Outlier threshold: {threshold:.3f}")
    print(f"    Kept {np.sum(mask)}/{len(mask)} points")
    
    return points3d[mask], mask


# ============================================================================
# BOUNDARY ESTIMATION
# ============================================================================

def estimate_boundary(points3d, z_threshold_sigma=2.0):
    """
    Estimate object boundary via 2D convex hull of points near the object.
    
    Strategy:
    1. Filter points based on Z-depth: keep points within ±z_threshold_sigma
       of the mean Z. This removes points that are clearly too close/far.
    2. Project filtered 3D points to XY plane
    3. Compute convex hull border in XY
    4. This gives the 2D projection of object extent
    
    Args:
        points3d: Nx3 array of 3D points
        z_threshold_sigma: How many std devs from mean Z to include
    
    Returns:
        boundary_hull: scipy.spatial.ConvexHull object
        boundary_points: Vertices of boundary (Mx2)
    """
    z_values = points3d[:, 2]
    mean_z = np.mean(z_values)
    std_z = np.std(z_values)
    
    z_min = mean_z - z_threshold_sigma * std_z
    z_max = mean_z + z_threshold_sigma * std_z
    
    mask = (z_values >= z_min) & (z_values <= z_max)
    boundary_points_3d = points3d[mask]
    
    print(f"  Boundary filtering: Z in [{z_min:.3f}, {z_max:.3f}]")
    print(f"  Selected {np.sum(mask)}/{len(mask)} points for boundary")
    
    if len(boundary_points_3d) < 3:
        print(f"  ERROR: Not enough points for boundary convex hull")
        return None, None
    
    # 2D convex hull in XY plane
    xy_points = boundary_points_3d[:, :2]
    hull = ConvexHull(xy_points)
    
    return hull, boundary_points_3d


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_reconstruction(images, all_points3d, camera_poses, K, 
                             hull=None, boundary_points=None,
                             output_file='sfm_reconstruction.png'):
    """
    Create 3D visualization with point cloud, camera trajectory, and boundary.
    
    Args:
        images: List of RGB images (for reference)
        all_points3d: Nx3 array of all triangulated 3D points
        camera_poses: List of (R, t) tuples for each camera
        K: Camera intrinsic matrix
        hull: ConvexHull object for boundary
        boundary_points: Points used for boundary
        output_file: Output filename for visualization
    """
    fig = plt.figure(figsize=(15, 5))
    
    # ---- Plot 1: 3D Reconstruction ----
    ax1 = fig.add_subplot(131, projection='3d')
    
    if len(all_points3d) > 0:
        # Color points by depth (Z coordinate)
        scatter = ax1.scatter(all_points3d[:, 0], all_points3d[:, 1], 
                             all_points3d[:, 2], c=all_points3d[:, 2],
                             cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Depth (Z)')
    
    # Draw camera positions
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']
    camera_positions = []
    
    # First camera at origin
    camera_positions.append(np.array([0.0, 0.0, 0.0]))
    ax1.scatter(*camera_positions[0], c='red', s=100, marker='s', 
               label='Camera 0', depthshade=False)
    
    # Subsequent cameras - accumulate pose
    current_pos = np.array([0.0, 0.0, 0.0])
    for i, (R, t) in enumerate(camera_poses):
        # World position of camera: -R^T @ t
        current_pos = current_pos - R.T @ t.flatten()
        camera_positions.append(current_pos.copy())
        color = colors[(i + 1) % len(colors)]
        ax1.scatter(*current_pos, c=color, s=100, marker='s', 
                   label=f'Camera {i+1}', depthshade=False)
    
    # Draw camera trajectory - convert to numpy array
    camera_trajectory = np.array(camera_positions, dtype=np.float64)
    ax1.plot(camera_trajectory[:, 0], camera_trajectory[:, 1], 
            camera_trajectory[:, 2], 'k--', linewidth=1, label='Trajectory')
    
    # Draw boundary if available
    if hull is not None and boundary_points is not None:
        boundary_vertices = boundary_points[hull.vertices]
        ax1.scatter(boundary_vertices[:, 0], boundary_vertices[:, 1],
                   boundary_points[0, 2] * np.ones(len(boundary_vertices)),
                   c='yellow', s=50, marker='^', label='Boundary', 
                   depthshade=False)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Reconstruction ({len(all_points3d)} points)')
    ax1.legend(fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # ---- Plot 2: XY Projected Boundary ----
    ax2 = fig.add_subplot(132)
    if len(all_points3d) > 0:
        ax2.scatter(all_points3d[:, 0], all_points3d[:, 1], s=10, 
                   alpha=0.3, c='blue', label='All 3D points')
    
    if hull is not None and boundary_points is not None:
        for simplex in hull.simplices:
            ax2.plot(boundary_points[simplex, 0], 
                    boundary_points[simplex, 1], 'r-', linewidth=2)
        boundary_vertices = boundary_points[hull.vertices]
        ax2.scatter(boundary_vertices[:, 0], boundary_vertices[:, 1],
                   c='yellow', s=100, marker='^', label='Boundary vertices',
                   zorder=5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Object Boundary (XY Projection)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete Structure from Motion reconstruction."""
    
    print("=" * 70)
    print("STRUCTURE FROM MOTION - COMPLETE 3D RECONSTRUCTION")
    print("=" * 70)
    
    # ---- Load Images ----
    print("\n[1] Loading images...")
    image_files = sorted(glob.glob('IMG_5732.jpg')) + sorted(glob.glob('IMG_5733.jpg')) + sorted(glob.glob('IMG_5735.jpg')) + sorted(glob.glob('IMG_5734.jpg'))[:5]  # Use up to 5 images
    print(f"    Found {len(image_files)} image files: {image_files}")
    
    images, gray_images, K = load_images(image_files, undistort=True)
    print(f"    Loaded: {len(images)} images")
    print(f"    Image shape: {images[0].shape}")
    print(f"    Camera matrix K:\n{K}")
    
    # ---- Feature Detection ----
    print("\n[2] Detecting SIFT features...")
    all_keypoints = []
    all_descriptors = []
    
    for i, gray in enumerate(gray_images):
        kpts, descs = detect_sift_features(gray, n_features=5000)
        all_keypoints.append(kpts)
        all_descriptors.append(descs)
        print(f"    Image {i}: {len(kpts)} keypoints")
    
    # ---- Sequential Structure from Motion ----
    print("\n[3] Sequential Structure from Motion...")
    all_points3d = []
    camera_poses = []  # Relative poses (R, t)
    
    # Cumulative camera positions for visualization
    current_R = np.eye(3)
    current_t = np.zeros((3, 1))
    
    # Process consecutive image pairs
    for pair_idx in range(len(images) - 1):
        i1, i2 = pair_idx, pair_idx + 1
        print(f"\n  View pair {i1} -> {i2}:")
        
        # ---- Feature Matching ----
        print(f"    Matching features...")
        matches = match_features(all_descriptors[i1], all_descriptors[i2])
        
        if len(matches) < 8:
            print(f"    Skipping pair: too few matches")
            continue
        
        pts1, pts2 = extract_matched_points(all_keypoints[i1], 
                                            all_keypoints[i2], matches)
        
        # ---- Epipolar Geometry ----
        print(f"    Computing Fundamental matrix...")
        F, pts1_inlier, pts2_inlier = compute_fundamental_matrix(pts1, pts2)
        
        if F is None:
            continue
        
        # Essential matrix
        E = compute_essential_matrix(F, K)
        
        # ---- Pose Recovery ----
        print(f"    Recovering camera pose...")
        R_relative, t_relative, pose_mask = recover_pose(E, pts1_inlier, 
                                                          pts2_inlier, K)
        
        # Don't re-filter with pose_mask - recoverPose already solved cheirality
        # internally and selected the correct solution. Use all inlier points for triangulation
        pts1_pose = pts1_inlier
        pts2_pose = pts2_inlier
        
        # ---- Triangulation ----
        print(f"    Triangulating 3D points...")
        
        # Projection matrices - construct carefully as contiguous float32
        I = np.eye(3, dtype=np.float32)
        zeros = np.zeros((3, 1), dtype=np.float32)
        P1 = np.ascontiguousarray(K @ np.column_stack([I, zeros]), dtype=np.float32)
        P2 = np.ascontiguousarray(K @ np.column_stack([np.float32(R_relative), 
                                                        np.float32(t_relative)]), 
                                  dtype=np.float32)
        
        # Triangulate points (don't pass mask since we already filtered above)
        points3d = triangulate_points(P1, P2, pts1_pose, pts2_pose)
        
        # Filter by depth
        points3d_filtered, depth_mask = filter_points_by_depth(points3d)
        print(f"    Triangulated: {len(points3d)} points, "
              f"kept {len(points3d_filtered)} after depth filtering")
        
        if len(points3d_filtered) > 0:
            # Filter outliers
            points3d_clean, outlier_mask = filter_outliers_statistical(
                points3d_filtered, n_sigma=3.0)
            all_points3d.extend(points3d_clean)
        
        camera_poses.append((R_relative, t_relative))
    
    # Convert to numpy array
    all_points3d = np.array(all_points3d)
    print(f"\n  Total 3D points: {len(all_points3d)}")
    
    # ---- Boundary Estimation ----
    print("\n[4] Estimating object boundary...")
    if len(all_points3d) >= 3:
        hull, boundary_points = estimate_boundary(all_points3d, 
                                                   z_threshold_sigma=2.0)
        if hull is not None:
            print(f"    Boundary convex hull: {len(hull.vertices)} vertices")
    else:
        hull, boundary_points = None, None
    
    # ---- Visualization ----
    print("\n[5] Creating visualization...")
    visualize_reconstruction(images, all_points3d, camera_poses, K,
                            hull=hull, boundary_points=boundary_points)
    
    # ---- Summary Statistics ----
    print("\n" + "=" * 70)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 70)
    print(f"Images processed: {len(images)}")
    print(f"Total 3D points: {len(all_points3d)}")
    if len(all_points3d) > 0:
        print(f"X range: {all_points3d[:, 0].min():.3f} to "
              f"{all_points3d[:, 0].max():.3f}")
        print(f"Y range: {all_points3d[:, 1].min():.3f} to "
              f"{all_points3d[:, 1].max():.3f}")
        print(f"Z range: {all_points3d[:, 2].min():.3f} to "
              f"{all_points3d[:, 2].max():.3f}")
    if hull is not None:
        print(f"Boundary vertices: {len(hull.vertices)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
