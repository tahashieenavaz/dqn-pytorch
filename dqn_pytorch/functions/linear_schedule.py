def linear_schedule(
    start_epsilon: float, end_epsilon: float, duration: int, t: int
) -> float:
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * t + start_epsilon, end_epsilon)
