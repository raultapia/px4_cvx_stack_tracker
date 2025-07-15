import numpy as np
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("tracker")
logger.setLevel(logging.DEBUG)
INCLUDE_ELAPSED_TIME_IN_DEBUG = False
INCLUDE_CONTAINER_SIZES_IN_DEBUG = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("\033[1;37mTRACKER %(levelname)s\033[0m: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def elapsed(func):
    def wrapper(*args, **kwargs):
        if INCLUDE_ELAPSED_TIME_IN_DEBUG:
            start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if INCLUDE_ELAPSED_TIME_IN_DEBUG:
            elapsed_time = time.perf_counter() - start_time
            logger.debug(f"[{func.__name__}] Elapsed time: {elapsed_time:.6f} seconds")
        return result
    return wrapper


class KalmanFilter:
    def __init__(
        self,
        init_state: np.ndarray,
        init_cov: np.ndarray = np.diag([1.0] * 3 + [5.0] * 3),
        process_cov: np.ndarray = np.diag([0.05] * 3 + [1.0] * 3),
        meas_cov: np.ndarray = np.diag([0.1] * 3),
        vel_meas_cov: np.ndarray = np.diag([10.0] * 3)
    ):
        self.state = init_state.copy()          # 6x1: [x, y, z, vx, vy, vz]
        self.cov = init_cov.copy()              # 6x6 P matrix
        self.Q = process_cov.copy()             # 6x6 process noise
        self.R = meas_cov.copy()                # 3x3 position measurement noise
        self.Rv = vel_meas_cov.copy()           # 3x3 velocity measurement noise

        self._A_base = np.eye(6)
        self._A_base[0, 3] = 1.0
        self._A_base[1, 4] = 1.0
        self._A_base[2, 5] = 1.0

        logger.debug("KalmanFilter initialized")

    def predict(self, dt: float, save: bool):
        A = self._A_base.copy()
        A[0, 3] = A[1, 4] = A[2, 5] = dt
        if save:
            self.state = A @ self.state
            self.cov = A @ self.cov @ A.T + self.Q
            return self.state
        else:
            return A @ self.state

    def update(self, z_pos: np.ndarray, v_meas: np.ndarray):
        z_aug = np.hstack([z_pos, v_meas])
        H = np.eye(6)
        R_aug = np.block([[self.R, np.zeros((3, 3))], [np.zeros((3, 3)), self.Rv]])
        y = z_aug - H @ self.state
        S = H @ self.cov @ H.T + R_aug
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.cov = (np.eye(6) - K @ H) @ self.cov


class MultiObjectTracker:
    @dataclass
    class TrackedObject:
        kf: KalmanFilter
        id: int = -1
        category: int = -1
        axes: np.ndarray = field(default_factory=lambda: np.zeros(3))
        last_t: float = 0
        last_z: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def euclideandist(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def match(a, b):
        if not a.category == b['category']:
            return float('inf')
        if MultiObjectTracker.euclideandist(a.kf.state[0:3], b['center']) > 1.0:
            return float('inf')
        return MultiObjectTracker.euclideandist(a.kf.state[0:3], b['center'])

    def __init__(self):
        self.trackers = []
        self.t = time.time_ns() / 1e9
        self.ros = False
        logger.debug("MultiObjectTracker initialized")

    @elapsed
    def estimate(self):
        outputs = []
        dt = time.time_ns() / 1e9 - self.t
        for t in self.trackers:
            pred = t.kf.predict(dt, save=False)
            outputs.append({'center': pred[0:3].copy(), 'category': t.category, 'axes': t.axes})

        # Publish
        if self.ros:
            self.publishRosMessage("estimate")

        if INCLUDE_CONTAINER_SIZES_IN_DEBUG:
            logger.debug(f"[estimate] trackers size: {len(self.trackers)}")

        return outputs

    @elapsed
    def track(self, detections):
        if len(detections) > 50:
            logging.warning(f'[track] called with {len(detections)} detections. This values is too high.')

        # Prediction
        dt = time.time_ns() / 1e9 - self.t
        for t in self.trackers:
            t.kf.predict(dt, save=True)
        self.t = time.time_ns() / 1e9

        # Remove missing tracks
        elapsed_time_to_remove = 1
        if logger.isEnabledFor(logging.DEBUG):
            removed_trackers = [t for t in self.trackers if self.t - t.last_t >= elapsed_time_to_remove]
            for t in removed_trackers:
                logger.debug(f"Removed tracked object with ID {t.id}")
        self.trackers = [t for t in self.trackers if self.t - t.last_t < elapsed_time_to_remove]

        # Update
        for det in detections:
            distances = [float('inf')] * len(self.trackers)
            for i in range(len(self.trackers)):
                distances[i] = self.match(self.trackers[i], det)
            if any(d != float('inf') for d in distances):
                ibest = distances.index(min(distances))
                v_meas = (det['center'] - self.trackers[ibest].last_z) / dt if dt > 0 else np.zeros(3)
                self.trackers[ibest].kf.update(det['center'], v_meas)
                self.trackers[ibest].last_z = det['center'].copy()
                self.trackers[ibest].last_t = self.t
            else:
                init_state = np.hstack([det['center'], np.zeros(3)])
                kf = KalmanFilter(init_state)
                self.trackers.append(
                    MultiObjectTracker.TrackedObject(
                        kf=kf,
                        id=str(time.time_ns()),
                        category=det['category'],
                        axes=det['axes'],
                        last_t=self.t,
                        last_z=det['center'].copy()
                    )
                )
                logger.debug(f"New tracked object added with ID {self.trackers[-1].id}")

        # Publish
        if self.ros:
            self.publishRosMessage("track")

        if INCLUDE_CONTAINER_SIZES_IN_DEBUG:
            logger.debug(f"[track] detections size: {len(detections)}")
            logger.debug(f"[track] trackers size: {len(self.trackers)}")

    @elapsed
    def enableRosPublishers(self, node, ros_version=2):
        from std_msgs.msg import String
        self.node = node
        self.pub = {}
        if ros_version == 1:
            import rospy
            self.pub["estimate"] = rospy.Publisher('/px4_cvx_stack_tracker/estimate', String, 10)
            self.pub["track"] = rospy.Publisher('/px4_cvx_stack_tracker/track', String, 10)
        if ros_version == 2:
            self.pub["estimate"] = node.create_publisher(String, '/px4_cvx_stack_tracker/estimate', 10)
            self.pub["track"] = node.create_publisher(String, '/px4_cvx_stack_tracker/track', 10)
        self.ros = True

    @elapsed
    def publishRosMessage(self, topic):
        from std_msgs.msg import String
        msg = String()
        msg.data = self.encodeRosMessage(topic)
        self.pub[topic].publish(msg)

    def encodeRosMessage(self, topic):
        sep = "  "
        m = ""
        for t in self.trackers:
            m += f'{3*sep}{{\n{4*sep}"id": {t.id},\n{4*sep}"category": {t.category},\n{4*sep}"axes": {t.axes.tolist()},\n{4*sep}"position": {t.kf.state[0:3].tolist()},\n{4*sep}"velocity": {t.kf.state[3:6].tolist()}\n{3*sep}}},\n'
        m = m[:-2] if m else ""

        return f'''{{
{sep}"{topic}": {{
{sep}{sep}"time": {self.t},
{sep}{sep}"menuitem": [
{m}
{sep}{sep}]
{sep}}}
}}'''


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    logger.warning("You are running a minimum working example of px4_cvx_stack_tracker")

    tracker = MultiObjectTracker()

    detections = [
        {'center': np.array([5.0, 8.0, 0.0]), 'category': 2, 'axes': np.array([1.0, 2.0, 3.0])},
        {'center': np.array([8.0, 3.0, 2.0]), 'category': 6, 'axes': np.array([3.0, 1.0, 2.0])},
        {'center': np.array([1.0, 6.0, 3.0]), 'category': 8, 'axes': np.array([2.0, 3.0, 1.0])}
    ]

    tracker.track(detections)

    times = []
    positions = {i: {'x': [], 'y': [], 'z': []} for i in range(len(detections))}
    real_positions = {i: {'x': [], 'y': [], 'z': []} for i in range(len(detections))}
    update_times = []

    cnt = 0
    for t in np.arange(0.01, 5.00, 0.01):
        cnt += 1
        time.sleep(0.01)

        for k in range(len(detections)):
            detections[k]['center'][0] += 0.010 + np.random.normal(0, 0.01)
            detections[k]['center'][1] += 0.020 + np.random.normal(0, 0.01)
            detections[k]['center'][2] += 0.001 + np.random.normal(0, 0.01)

        objects = tracker.estimate()

        if cnt == 20:
            if t < 2.5:
                tracker.track(detections)
            else:
                tracker.track(detections[0:2])  # Simulate one missing obstacle
            update_times.append(t)
            cnt = 0

        times.append(t)
        for i in range(len(detections)):
            if i < len(objects):
                positions[i]['x'].append(objects[i]['center'][0])
                positions[i]['y'].append(objects[i]['center'][1])
                positions[i]['z'].append(objects[i]['center'][2])
            else:
                positions[i]['x'].append(np.nan)
                positions[i]['y'].append(np.nan)
                positions[i]['z'].append(np.nan)

            real_positions[i]['x'].append(detections[i]['center'][0])
            real_positions[i]['y'].append(detections[i]['center'][1])
            real_positions[i]['z'].append(detections[i]['center'][2])

    fig, axs = plt.subplots(3, 1, sharex=True)
    for i in range(len(detections)):
        axs[0].plot(times, positions[i]['x'], color=f'C{i}', label=f'Obs {i+1} Pred X')
        axs[0].plot(times, real_positions[i]['x'], '--', color=f'C{i}', label=f'Obs {i+1} Meas X')
        axs[1].plot(times, positions[i]['y'], color=f'C{i}', label=f'Obs {i+1} Pred Y')
        axs[1].plot(times, real_positions[i]['y'], '--', color=f'C{i}', label=f'Obs {i+1} Meas Y')
        axs[2].plot(times, positions[i]['z'], color=f'C{i}', label=f'Obs {i+1} Pred Z')
        axs[2].plot(times, real_positions[i]['z'], '--', color=f'C{i}', label=f'Obs {i+1} Meas Z')

    axs[0].set_ylabel('X')
    axs[0].grid()
    axs[1].set_ylabel('Y')
    axs[1].grid()
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z')
    axs[2].grid()

    for ut in update_times:
        for ax in axs:
            ax.axvline(x=ut, color='r', linestyle='-', linewidth=1)

    plt.show()
