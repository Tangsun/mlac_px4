import numpy as np


def generate_time_windows(timeline, duration, step=None, min_points=2):
    """
    Yield rolling windows over the provided timeline.
    Returns tuples of (start_time, end_time, start_idx, end_idx).
    """
    times = np.asarray(timeline)
    if times.ndim != 1 or times.size < min_points:
        raise ValueError("timeline must be a 1D array with at least two samples.")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    if step is None:
        step = duration
    if step <= 0:
        raise ValueError("step must be positive.")

    t_start = times[0]
    t_final = times[-1]
    current = t_start

    while current < t_final:
        end_time = min(current + duration, t_final)
        start_idx = np.searchsorted(times, current, side='left')
        end_idx = np.searchsorted(times, end_time, side='right')

        if end_idx - start_idx >= min_points:
            yield current, end_time, start_idx, end_idx

        current += step
