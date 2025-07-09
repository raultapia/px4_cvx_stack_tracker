# PX4 CVX Stack Tracker

[Raul Tapia](https://raultapia.com) @ University of Seville

This is a lightweight Python library for 3D multi-object tracking using Kalman Filters. It includes tools for backprojecting 2D image coordinates into 3D space using camera intrinsics and tracking multiple objects over time with a simple motion model. This library is part of the `px4_cvx_stack_tracker`.

## Camera Backprojection

The `Camera` class is used to backproject 2D pixel coordinates into 3D space using known depth and extrinsic calibration:

```python
from camera import Camera
import numpy as np

camera = Camera(fx=800, fy=800, cx=640, cy=360)
R_wc = np.eye(3)
t_wc = np.zeros(3)

u, v, d = 640, 360, 5.0  # Pixel coordinates (u, v) and depth d
point_3d = camera.backproject(u, v, d, R_wc, t_wc)
print("3D Point:", point_3d)
```

## Multi-Object Tracking

The `MultiObjectTracker` class manages the tracking of multiple moving objects using simple constant velocity models and Euclidean matching:

```python
from tracker import MultiObjectTracker
import numpy as np

tracker = MultiObjectTracker()

detections = [
    {'center': np.array([5.0, 8.0, 0.0]), 'category': 2, 'axes': np.array([1.0, 1.0, 1.0])},
    {'center': np.array([8.0, 3.0, 2.0]), 'category': 6, 'axes': np.array([1.0, 1.0, 1.0])}
]

tracker.track(detections) # Track objects with detections

[...]

predictions = tracker.estimate() # Predict positions after time delta
for obj in predictions:
    print(f"Predicted Object: Center={obj['center']}, Category={obj['category']}")
```
