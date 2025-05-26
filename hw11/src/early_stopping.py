class EarlyStopper:
    """Early stopper to stop training when a certain metric does not improve for a certain number
    of epochs.

    :param patience: Number of epochs to wait before stopping if the metric does not improve.
    :param mode: One of "min" or "max". Use 'min' for loss-like and 'max' for accuracy-like metrics
    """

    def __init__(self, patience, mode="min"):
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be either 'min' or 'max', got {mode}")
        self.mode = mode
        self.patience = patience
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def _is_better(self, val1, val2):
        if self.mode == "min":
            return val1 < val2
        elif self.mode == "max":
            return val1 > val2

    def update(self, value) -> bool:
        """Update the early stopper with a new value. Call this after each epoch. Returns True if
        training should stop.
        """
        if self._is_better(value, self.best_value):
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "patience": self.patience,
            "best_value": self.best_value,
            "counter": self.counter,
        }

    def load_state_dict(self, sate_dict: dict) -> "EarlyStopper":
        self.best_value = sate_dict["best_value"]
        self.counter = sate_dict["counter"]
        self.mode = sate_dict["mode"]
        self.patience = sate_dict["patience"]
        return self
