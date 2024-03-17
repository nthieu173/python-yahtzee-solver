import pickle
import time
from bisect import bisect_left
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable

import numpy as np
import numpy.typing as npt

from dice_states import (
    DICE_STATES,
    KEEP_ACTIONS,
    NUM_DICE_STATES,
    load_transition_function,
)
from score_states import ScoreState

ScoreDiceStateRollRewardType = npt.NDArray[np.float64]

SCORE_DICE_STATE_ROLL_REWARD_FILE = "score_dice_state_roll_reward.npy"

NUM_ROLLS = 3


TRANSITION_FUNCTION = load_transition_function()

KEEP_NONE_ACTION = (0, 0, 0, 0, 0, 0)
KEEP_NON_ACTION_INDEX = KEEP_ACTIONS.index(KEEP_NONE_ACTION)

# The probablity of rolling any state by rerolling all dices
FIRST_ROLL_PROBABILITY = TRANSITION_FUNCTION[0, KEEP_NON_ACTION_INDEX, :]


class LayerScoreDiceStateRollReward(Mapping[int, npt.NDArray[np.float64]]):
    """
    A mapping from ScoreState.as_int() to its reward for each DiceState and Reroll.

    The values are stored in a shared memory, so it can be shared between processes.
    """

    def __init__(
        self,
        keys: Iterable[int],
        shared_memory_name: str | None = None,
    ):
        self.sorted_keys = sorted(keys)
        temp_values = np.zeros((len(self.sorted_keys), NUM_ROLLS, NUM_DICE_STATES))
        self.shared_memory = SharedMemory(
            name=shared_memory_name,
            create=shared_memory_name is None,
            size=temp_values.nbytes,
        )
        del temp_values
        self._values = np.frombuffer(
            self.shared_memory.buf,
            dtype=np.float64,
        ).reshape((len(self.sorted_keys), NUM_ROLLS, NUM_DICE_STATES))

    def __index(self, key: int) -> int:
        index = bisect_left(self.sorted_keys, key)
        if index != len(self.sorted_keys) and self.sorted_keys[index] == key:
            return index
        raise KeyError(key)

    def __getitem__(self, key: int) -> npt.NDArray[np.float64]:
        return self._values[self.__index(key)]

    def __iter__(self):
        return iter(self.sorted_keys)

    def __len__(self):
        return len(self.sorted_keys)


def calculate_score_state_reward():
    """
    Calculate the reward of each ScoreState.
    """

    print("Exploring terminal states...")
    terminal_states = ScoreState.get_all_terminal_state()
    exploration_set: set[ScoreState] = set()

    terminal_state_keys = []
    for terminal_state in terminal_states:
        terminal_state_keys.append(terminal_state.as_int())
        # Terminal states have no reward, so it's automatically explored
        for parent in terminal_state.parent_states():
            exploration_set.add(parent)

    # The previous layer reward is readonly, so we can share it between processes
    previous_layer_reward = LayerScoreDiceStateRollReward(terminal_state_keys)

    print("Exploring non-terminal states...")

    layer_num = 0
    with ProcessPoolExecutor(max_workers=16) as executor:
        while (num_states_in_layer := len(exploration_set)) > 0:
            new_exploration_set: set[ScoreState] = set()
            current_layer_reward = LayerScoreDiceStateRollReward(
                map(lambda x: x.as_int(), exploration_set),
            )

            previous_layer_state_ints_repeated_iter = iter(
                lambda: previous_layer_reward.sorted_keys, None
            )

            shared_memory_name_repeated_iter = iter(
                lambda: previous_layer_reward.shared_memory.name, None
            )

            count = 0
            start_time = time.monotonic()

            score_state: ScoreState
            score_state_reward: npt.NDArray[np.float64]
            for score_state, score_state_reward in executor.map(
                parrallelizable_calculate_score_state_reward,
                zip(
                    exploration_set,
                    previous_layer_state_ints_repeated_iter,
                    shared_memory_name_repeated_iter,
                ),
                chunksize=512,
            ):
                count += 1
                elapsed_time = time.monotonic() - start_time
                print!(
                    f"Evaluating layer {layer_num}: {count}/{num_states_in_layer} states. "
                    f"ETA: {(num_states_in_layer - count) * elapsed_time / count / 3600:.2f} hours",
                    end="\r",
                );
                std::io::stdout().flush().unwrap();

                current_layer_reward[score_state.as_int()][:, :] = score_state_reward

                for parent in score_state.parent_states():
                    new_exploration_set.add(parent)

            with open(
                f"score_dice_state_roll_reward_layer_{layer_num}.pickle", "wb"
            ) as f:
                pickle.dump(current_layer_reward, f)

            layer_num += 1
            del previous_layer_reward._values
            previous_layer_reward.shared_memory.close()
            previous_layer_reward.shared_memory.unlink()
            del previous_layer_reward
            previous_layer_reward = current_layer_reward
            exploration_set = new_exploration_set

    del previous_layer_reward._values
    previous_layer_reward.shared_memory.close()
    previous_layer_reward.shared_memory.unlink()
    del previous_layer_reward


CombinedParams = tuple[ScoreState, list[int], str]


def parrallelizable_calculate_score_state_reward(combined_params: CombinedParams):
    """
    Parallelizable version of calculate_score_state_reward.
    """
    score_state, previous_layer_state_ints, shared_memory_name = combined_params
    previous_layer_reward = LayerScoreDiceStateRollReward(
        previous_layer_state_ints, shared_memory_name
    )
    result = calculate_state_reward(score_state, previous_layer_reward)
    del previous_layer_reward._values
    previous_layer_reward.shared_memory.close()
    return result


def calculate_state_reward(
    score_state, previous_layer_reward
) -> tuple[ScoreState, npt.NDArray[np.float64]]:
    """
    Calculate the reward of a ScoreState, and return it as a numpy array
    of shape (NUM_ROLLS, NUM_DICE_STATES).
    """
    score_state_reward = np.zeros((NUM_ROLLS, NUM_DICE_STATES))
    # 0 reroll, the reward is the
    # Reward(ScoreState, DiceState, ScoreAction)
    # + Sum of (
    #   ProbFirstRoll * Reward(ChildScoreState, FirstRollDiceState, Reroll=2)
    # ) over all first roll dice states
    # Maximize over the possible actions to get
    # Reward(ScoreState, DiceState, Reroll=0)
    score_actions = score_state.possible_actions()
    for dice_state_index, dice_state in enumerate(DICE_STATES):
        max_reward = 0
        for score_action in score_actions:
            action_reward = score_state.reward(score_action, dice_state)
            child_score_state = score_state.apply_action(score_action, dice_state)
            child_reward = FIRST_ROLL_PROBABILITY.dot(
                previous_layer_reward[child_score_state.as_int()][2, :]
            )
            max_reward = max(max_reward, action_reward + child_reward)
        score_state_reward[0, dice_state_index] = max_reward

    # 1 and 2 reroll, the reward is the
    # Sum of (
    #   TransitionProbability(DiceState, KeepAction, ToDiceState)
    #   * Reward(ScoreState, ToDiceState, Reroll - 1)
    # ) over all ToDiceStates

    # Maximize over the possible KeepAction to get
    # Reward(ScoreState, DiceState, Reroll)

    for reroll in range(1, NUM_ROLLS):
        for dice_state_index in range(NUM_DICE_STATES):
            max_reward = 0
            for keep_action in KEEP_ACTIONS:
                keep_reward = TRANSITION_FUNCTION[
                    dice_state_index, sorted_index(KEEP_ACTIONS, keep_action), :
                ].dot(score_state_reward[reroll - 1, :])
                max_reward = max(max_reward, keep_reward)
            score_state_reward[reroll, dice_state_index] = max_reward

    return score_state, score_state_reward


def sorted_index(sorted_list: list[int], value: int) -> int:
    """
    Return the index of value in sorted_list.
    If value is not in sorted_list, return the index where value should be inserted.
    """
    index = bisect_left(sorted_list, value)
    if index != len(sorted_list) and sorted_list[index] == value:
        return index
    raise ValueError(f"{value} not in list")


if __name__ == "__main__":
    calculate_score_state_reward()
