from __future__ import annotations

from typing import Any

import numpy as np

from test.GraphWalkTest import GraphWalkTest, to_categorical


class GraphWalkTestCI(GraphWalkTest):
    """Harder GraphWalk variant for CI experiments.

    Keeps the same graph walk dynamics as GraphWalkTest, but emits a more
    ambiguous observation by mixing several recent states with configurable
    gains and decay. The default GraphWalkTest remains unchanged.
    """

    history_length: int
    decay_rate: float
    current_gain: float
    history_gain: float
    noise_intensity: float
    normalize_observation: bool
    skip_current_duplicates: bool
    state_history: list[int]

    def __init__(
        self,
        time_delay: int,
        filename: str = "graph.dot",
        *,
        history_length: int = 3,
        decay_rate: float = 0.0,
        current_gain: float = 1.0,
        history_gain: float = 1.0,
        noise_intensity: float = 0.0,
        normalize_observation: bool = False,
        skip_current_duplicates: bool = True,
    ) -> None:
        self.history_length = max(1, int(history_length))
        self.decay_rate = float(decay_rate)
        self.current_gain = float(current_gain)
        self.history_gain = float(history_gain)
        self.noise_intensity = float(noise_intensity)
        self.normalize_observation = bool(normalize_observation)
        self.skip_current_duplicates = bool(skip_current_duplicates)
        super().__init__(time_delay=time_delay, filename=filename)
        self.state_history = [int(self.output_class)]

    def switch_dataset(self, filename: str) -> None:
        super().switch_dataset(filename)
        self.state_history = [int(self.output_class)]

    def _push_state(self, state: int) -> None:
        self.state_history.insert(0, int(state))
        if len(self.state_history) > self.history_length:
            del self.state_history[self.history_length:]

    def getInput(self, reset: bool = False) -> np.ndarray[Any, Any]:
        if reset:
            self.time_counter = 0
            self.state_history = [int(self.output_class)]

        update = self.updateTimeDelay()
        if update:
            self.previous_output_class = self.output_class
            self.output_class = int(np.random.choice(self.output_size, 1, p=self.a[self.output_class])[0])
            self._push_state(int(self.output_class))

        if not self.state_history:
            self.state_history = [int(self.output_class)]

        observation = np.zeros(self.output_size, dtype=np.float64)
        current_state = int(self.state_history[0])
        for lag, state in enumerate(self.state_history):
            if lag > 0 and self.skip_current_duplicates and int(state) == current_state:
                continue
            gain = self.current_gain if lag == 0 else self.history_gain
            decay = np.exp(-self.decay_rate * (self.time_counter + lag * self.time_delay))
            observation += to_categorical(int(state), self.output_size) * gain * decay

        if self.noise_intensity > 0.0:
            observation += np.random.randn(self.output_size) * self.noise_intensity

        if self.normalize_observation:
            total = float(np.sum(np.abs(observation)))
            if total > 1e-12:
                observation = observation / total

        return observation
