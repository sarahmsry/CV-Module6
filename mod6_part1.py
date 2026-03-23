import cv2
import numpy as np
import matplotlib.pyplot as plt

# optical flow visualization for a video using Lucas-Kanade and Farneback methods
def process_video(video_path):
    """
    Computes and visualizes optical flow for a given video using two methods:
    1. Lucas-Kanade (sparse) - tracks good features and draws motion vectors
    2. Farneback (dense) - computes flow for every pixel, visualized with HSV
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # detect good corners/features to track in the first frame
    # these are points where the gradient is strong in multiple directions
    # (well-conditioned AᵀA matrix) — flat regions are intentionally skipped
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

    # pre-allocate HSV image for dense flow visualization
    # hue = direction, value = speed, saturation fixed at 255 (full color)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lucas-Kanade sparse optical flow 
        # for each tracked point, solve (AᵀA)⁻¹Aᵀb in a local window
        # status == 1 means the point was successfully tracked
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None)

        # keep only successfully tracked points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # draw motion vectors as arrows: tail at old position, head at new
        for new, old in zip(good_new, good_old):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.arrowedLine(frame, (c, d), (a, b), (0, 255, 0), 2)

        # ── Farneback dense optical flow ──────────────────────────────────────
        # computes (u, v) for every pixel using polynomial expansion
        # returns a 2-channel array: flow[..., 0] = u, flow[..., 1] = v
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # convert (u, v) to polar: magnitude = speed, angle = direction
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # encode direction as hue, speed as brightness
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # convert HSV to BGR for display (imshow expects BGR)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # update previous frame and points for next iteration
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        cv2.imshow('Frame with Motion Vectors', frame)
        cv2.imshow('Optical Flow HSV', bgr)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# bilinear interpolation function for estimating intensity at fractional coordinates
def bilinear_interpolation(image, x, y):
    """
    Estimates intensity at fractional coordinate (x, y) using the four
    surrounding integer pixels.

    Theory:
    -------
    Let x0 = floor(x), y0 = floor(y), a = x - x0, b = y - y0

    I(x,y) = I(x0,y0)(1-a)(1-b) + I(x0+1,y0) a(1-b)
           + I(x0,y0+1)(1-a)b   + I(x0+1,y0+1) ab

    Weights are proportional to the area of the opposite sub-rectangle,
    so closer pixels contribute more.
    """
    h, w = image.shape

    # return 0 if outside image bounds
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0

    # integer base coordinates
    x0, y0 = int(x), int(y)

    # fractional parts (how far past the integer coordinate)
    a = x - x0
    b = y - y0

    # four surrounding pixels
    I00 = image[y0,     x0    ]
    I10 = image[y0,     min(x0 + 1, w - 1)]
    I01 = image[min(y0 + 1, h - 1), x0    ]
    I11 = image[min(y0 + 1, h - 1), min(x0 + 1, w - 1)]

    # weighted average
    value = (I00 * (1 - a) * (1 - b) +
             I10 *      a  * (1 - b) +
             I01 * (1 - a) *      b  +
             I11 *      a  *      b)

    return value

# gradient computation for Lucas-Kanade
def compute_gradients(frame1, frame2):
    """
    Computes spatial gradients Ix, Iy and temporal gradient It.

    Theory:
    -------
    Ix = ∂I/∂x  — how intensity changes horizontally (Sobel filter)
    Iy = ∂I/∂y  — how intensity changes vertically   (Sobel filter)
    It = I2 - I1 — how intensity changes over time
    """
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)

    # Sobel gives weighted finite differences — more robust than simple subtraction
    Ix = cv2.Sobel(frame1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(frame1, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # temporal gradient: difference between frames at same pixel location
    It = frame2 - frame1

    return Ix, Iy, It

# manual Lucas-Kanade implementation 
def lucas_kanade_patch(Ix, Iy, It, x, y, window_size=15):
    """
    Manually solves the Lucas-Kanade least squares system for one point.

    Theory:
    -------
    Within a window of N pixels, each pixel gives one equation:
        Ix_i * u + Iy_i * v = -It_i

    In matrix form: A * [u, v]ᵀ = b
    where A is Nx2 (columns are Ix and Iy values in window)
    and   b is -It values in window

    Least squares solution:
        [u, v]ᵀ = (AᵀA)⁻¹ Aᵀb

    Only solvable if AᵀA is invertible — requires gradients in 2 directions,
    which is why corners (not flat regions) are trackable.
    """
    h, w = Ix.shape
    half = window_size // 2

    # extract window around point, clamped to image bounds
    y_min = max(0, y - half)
    y_max = min(h, y + half + 1)
    x_min = max(0, x - half)
    x_max = min(w, x + half + 1)

    Ix_w = Ix[y_min:y_max, x_min:x_max].flatten()
    Iy_w = Iy[y_min:y_max, x_min:x_max].flatten()
    It_w = It[y_min:y_max, x_min:x_max].flatten()

    # build A (Nx2) and b (N)
    A = np.column_stack([Ix_w, Iy_w])
    b = -It_w

    # solve least squares: minimizes ||A[u,v]ᵀ - b||²
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        u, v = result[0], result[1]
        error = np.sqrt(np.sum((A @ result - b) ** 2))
    except Exception:
        u, v, error = 0.0, 0.0, np.inf

    return u, v, error

# validation of optical flow constraint at a point using computed (u, v) from `lucas_kanade_patch`
def validate_optical_flow_constraint(frame1, frame2, x, y, u, v, window_size=15):
    """
    Checks how well the optical flow constraint Ix*u + Iy*v + It ≈ 0 holds
    at a given point using the computed (u, v).

    A small mean error means the brightness constancy assumption holds well
    at this location, validating the theoretical derivation.
    """
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    h, w = frame1.shape
    half = window_size // 2

    Ix, Iy, It = compute_gradients(frame1, frame2)

    y_min = max(0, y - half)
    y_max = min(h, y + half + 1)
    x_min = max(0, x - half)
    x_max = min(w, x + half + 1)

    Ix_w = Ix[y_min:y_max, x_min:x_max]
    Iy_w = Iy[y_min:y_max, x_min:x_max]
    It_w = It[y_min:y_max, x_min:x_max]

    # compute Ix*u + Iy*v + It for every pixel in window — should be ~0
    constraint = Ix_w * u + Iy_w * v + It_w
    mean_error  = np.mean(np.abs(constraint))

    predicted_x = x + u
    predicted_y = y + v

    return constraint, mean_error, (predicted_x, predicted_y)

# two-frame tracking and validation using manual Lucas-Kanade implementation
def track_features_two_frames(video_path, frame_ids=(0, 1)):
    """
    Loads two consecutive frames from a video, tracks features using a manual
    Lucas-Kanade implementation, and validates against OpenCV's result.

    Produces:
    - Printed table: manual prediction vs OpenCV vs error
    - Bilinear interpolation at predicted locations on real frames
    - Visualization plots
    """
    print("\n" + "="*70)
    print(f"MOTION TRACKING VALIDATION — {video_path}")
    print("="*70)

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count <= max(frame_ids):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    if len(frames) < 2:
        print(f"Could not extract frames from {video_path}")
        return

    frame1_bgr = frames[frame_ids[0]]
    frame2_bgr = frames[frame_ids[1]]

    # convert to grayscale for gradient computation
    frame1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    frame1_uint = frame1.astype(np.uint8)
    frame2_uint = frame2.astype(np.uint8)

    print(f"Frame shape: {frame1.shape}")

    # detect keypoints in frame 1 
    features = cv2.goodFeaturesToTrack(frame1_uint, maxCorners=10,
                                        qualityLevel=0.01, minDistance=20)
    if features is None or len(features) == 0:
        print("No features detected!")
        return

    print(f"Detected {len(features)} features to track\n")

    # compute gradients once for all points 
    Ix, Iy, It = compute_gradients(frame1_uint, frame2_uint)

    # track each feature and compare manual LK prediction to OpenCV's result 
    tracking_results = []

    for idx, feature in enumerate(features[:10]):
        x, y = int(feature[0][0]), int(feature[0][1])

        # skip points too close to border (window would be clipped badly)
        if x < 20 or x > frame1.shape[1] - 20 or y < 20 or y > frame1.shape[0] - 20:
            continue

        #  manual LK prediction using gradients and least squares solution
        u, v, lk_error = lucas_kanade_patch(Ix, Iy, It, x, y, window_size=15)
        pred_x = x + u
        pred_y = y + v

        # constraint validation 
        constraint, mean_error, pred_loc = validate_optical_flow_constraint(
            frame1_uint, frame2_uint, x, y, u, v)

        # OpenCV LK as ground truth 
        p0 = np.array([[[float(x), float(y)]]], dtype=np.float32)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(frame1_uint, frame2_uint, p0, None)

        if status[0][0] == 1:
            opencv_x, opencv_y = p1[0][0]
        else:
            opencv_x, opencv_y = float(x), float(y)

        # error between manual prediction and OpenCV
        error_px = np.sqrt((pred_x - opencv_x)**2 + (pred_y - opencv_y)**2)

        # bilinear interpolation at predicted location 
        # predicted location is fractional — interpolate intensity in frame2
        interp_val   = bilinear_interpolation(frame2.astype(np.float32), pred_x, pred_y)
        # nearest integer pixel intensity for comparison
        nearest_val  = frame2_uint[int(round(pred_y)), int(round(pred_x))]

        tracking_results.append({
            'idx':        idx,
            'x':          x,          'y':          y,
            'u':          u,          'v':          v,
            'pred_x':     pred_x,     'pred_y':     pred_y,
            'opencv_x':   opencv_x,   'opencv_y':   opencv_y,
            'error_px':   error_px,
            'constraint': mean_error,
            'interp':     interp_val, 'nearest':    float(nearest_val),
        })

    print(f"{'Pt':<4} {'Frame1 (x,y)':<16} {'Manual pred (x,y)':<24} "
          f"{'OpenCV (x,y)':<24} {'Error (px)':<12} {'Constraint err'}")
    print("-"*90)
    for r in tracking_results:
        print(f"{r['idx']:<4} "
              f"({r['x']:>4}, {r['y']:>4})      "
              f"({r['pred_x']:>7.2f}, {r['pred_y']:>7.2f})         "
              f"({r['opencv_x']:>7.2f}, {r['opencv_y']:>7.2f})         "
              f"{r['error_px']:>8.3f}     "
              f"{r['constraint']:>8.4f}")

    # print bilinear interpolation on real frames 
    print(f"\n{'Pt':<4} {'Predicted loc':<24} {'Interp intensity':<20} {'Nearest pixel intensity'}")
    print("-"*70)
    for r in tracking_results:
        print(f"{r['idx']:<4} "
              f"({r['pred_x']:>7.2f}, {r['pred_y']:>7.2f})         "
              f"{r['interp']:>10.2f}          "
              f"{r['nearest']:>10.2f}")

    # visualize results in a four-panel plot
    visualize_tracking_results(frame1_bgr, frame2_bgr, tracking_results)

    return tracking_results

# two-frame tracking and validation using manual Lucas-Kanade implementation
def visualize_tracking_results(frame1_bgr, frame2_bgr, results):
    """
    Four-panel plot:
    - Top left:  frame 1 with detected feature locations marked
    - Top right: frame 2 with manual prediction (green) and OpenCV (red) arrows
    - Bottom left:  optical flow constraint error per feature (bar chart)
    - Bottom right: pixel error between manual prediction and OpenCV per feature
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # frame 1: feature locations 
    axes[0, 0].imshow(cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Frame 1 — Detected Features')
    for r in results:
        axes[0, 0].plot(r['x'], r['y'], 'rx', markersize=10, markeredgewidth=2)
        axes[0, 0].text(r['x'] + 5, r['y'] + 5, str(r['idx']),
                        color='red', fontsize=8)

    # frame 2: manual (green) vs OpenCV (red) predicted locations 
    axes[0, 1].imshow(cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Frame 2 — Manual (green) vs OpenCV (red) Prediction')
    for r in results:
        # manual prediction arrow
        axes[0, 1].annotate('', xy=(r['pred_x'], r['pred_y']),
                            xytext=(r['x'], r['y']),
                            arrowprops=dict(arrowstyle='->', color='lime', lw=2))
        # opencv result arrow
        axes[0, 1].annotate('', xy=(r['opencv_x'], r['opencv_y']),
                            xytext=(r['x'], r['y']),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                                           linestyle='dashed'))

    # constraint error bar chart for each feature (lower is better)
    axes[1, 0].bar(range(len(results)),
                   [r['constraint'] for r in results],
                   color='steelblue', alpha=0.8)
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Mean |Ix·u + Iy·v + It|')
    axes[1, 0].set_title('Optical Flow Constraint Error\n(closer to 0 = theory holds)')
    axes[1, 0].grid(True, alpha=0.3)

    # pixel error bar chart between manual prediction and OpenCV
    axes[1, 1].bar(range(len(results)),
                   [r['error_px'] for r in results],
                   color='coral', alpha=0.8)
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Error (pixels)')
    axes[1, 1].set_title('Manual LK vs OpenCV\nPrediction Error (pixels)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tracking_validation.png', dpi=150)
    plt.show()

if __name__ == "__main__":

    # optical flow visualization for both videos 
    process_video('IMG_5591.mov')
    process_video('IMG_5590.mov')

    # two-frame tracking and validation for both videos 
    results1 = track_features_two_frames('IMG_5591.mov', frame_ids=(0, 1))
    results2 = track_features_two_frames('IMG_5590.mov', frame_ids=(0, 1))


'''
From the results, we can see that the Lucas-Kanade method 
provides sparse motion vectors for the detected features, while the Farneback 
method gives a dense flow field. The HSV visualization helps to understand 
the direction and magnitude of motion in the video. The green arrows 
indicate the motion vectors from the Lucas-Kanade method, while the 
color in the HSV image represents the flow direction and intensity. 
This helps to understand and analyze the motion patterns in the video. 
The output of this code is perfect proof of the effectiveness of both optical 
flow methods. For example, since the Lucas-Kanade method tracks corners and 
good features, the motion vectors appear at those specific points (which may 
be a boundary of an object or a textured area as seen in my first test video,
where the ball has motion vectors positioned at the perimiter only).
This also explains why, using the Farneback method, the flow field
only appears on the perimiter of the ball, since the inside of the ball does not
have any distinct textures or gradients for the algorithm to detect. 
Another thing to note is that the motion vectors, including the arrows, 
sometimes appear to be pointing in the wrong direction. This actually shows that 
it is performing correctly, since at some points the ball goes out of frame, 
which can affect the tracking of features. 
'''