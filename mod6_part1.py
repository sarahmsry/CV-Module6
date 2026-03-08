import cv2
import numpy as np

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

    # Create HSV image for flow visualization
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        
        # Draw motion vectors
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]
        for new, old in zip(good_new, good_old):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.arrowedLine(frame, (c, d), (a, b), (0, 255, 0), 2)
        
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Visualize using HSV: hue = direction, value = magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)  # convert before showing
                
        # Display the results
        cv2.imshow('Frame with Motion Vectors', frame)
        cv2.imshow('Optical Flow HSV', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Keep windows open after video ends
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Processing video: {process_video('IMG_5590.mov')}")
    

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
