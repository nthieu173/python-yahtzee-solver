"""
Generate the transition function for Yahtzee dice states
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import factorial, fsum, prod

import numpy as np
import numpy.typing as npt

DiceStateType = tuple[int, int, int, int, int, int]
KeepActionType = tuple[int, int, int, int, int, int]
CombinedParamsType = tuple[DiceStateType, KeepActionType, DiceStateType]
TransitionFunctionType = npt.NDArray[np.float64]

# A state is a 5 tuple of integers representing the number of dice of each value.
DICE_STATES = []
for _a in range(7):
    for _b in range(7 - _a):
        for _c in range(7 - _a - _b):
            for _d in range(7 - _a - _b - _c):
                for _e in range(7 - _a - _b - _c - _d):
                    _f = 6 - _a - _b - _c - _d - _e
                    DICE_STATES.append((_a, _b, _c, _d, _e, _f))
DICE_STATES.sort()


TRANSITION_FUNCTION_FILE = "transition_function.npy"


NUM_DICE_STATES = len(DICE_STATES)

# An action is a 6 tuple of booleans representing which dice to keep.
KEEP_ACTIONS = []
for _a in (False, True):
    for _b in (False, True):
        for _c in (False, True):
            for _d in (False, True):
                for _e in (False, True):
                    for _f in (False, True):
                        KEEP_ACTIONS.append((_a, _b, _c, _d, _e, _f))

KEEP_ACTIONS.sort()

NUM_KEEP_ACTIONS = len(KEEP_ACTIONS)


def generate_transition_function() -> TransitionFunctionType:
    """
    Generate the transition function for Yahtzee.
    """
    transition_function = np.zeros((NUM_DICE_STATES, NUM_KEEP_ACTIONS, NUM_DICE_STATES))
    count = 0
    with ProcessPoolExecutor() as executor:
        params = [
            (i, j, k)
            for i in range(NUM_DICE_STATES)
            for j in range(NUM_KEEP_ACTIONS)
            for k in range(NUM_DICE_STATES)
        ]
        for (x, y, z), prob in zip(
            params,
            executor.map(
                transition_probabilities,
                [
                    (DICE_STATES[i], KEEP_ACTIONS[j], DICE_STATES[k])
                    for i, j, k in params
                ],
                chunksize=NUM_DICE_STATES * NUM_DICE_STATES,
            ),
        ):
            transition_function[x, y, z] = prob
            count += 1

    return transition_function


def transition_probabilities(
    params: CombinedParamsType,
) -> float:
    """
    Given a state, action, and next state, return the probability of transitioning from the
    given state to the given next state given the given action.
    """
    state, action, next_state = params

    # Get kept dices.
    kept = action_to_kept_tuple(state, action)

    # Get goal reroll dices.
    goal_reroll = tuple(map(lambda x: x[0] - x[1], zip(next_state, kept)))

    # If there are negative goal reroll dices, then the transition is impossible
    # because we are keeping more dice of a certain value than we are trying to get.
    if any(map(lambda x: x < 0, goal_reroll)):
        return 0.0
    if sum(goal_reroll) == 0:
        return 1.0

    # Get indices of goal reroll dices that are positive

    return probability_of_goal_roll(goal_reroll)


@lru_cache(maxsize=None)
def action_to_kept_tuple(state: DiceStateType, action: KeepActionType) -> DiceStateType:
    """
    Given a state and an action, return the state that results from keeping the dice
    specified by the action.

    For example, if state = (2, 3, 1, 0, 0, 0) and action = (1, 0, 1, 0, 0, 0), then
    action_to_kept_tuple(state, action) = (1, 1, 0, 0, 0, 0).
    """
    action_index = 0
    kept = [0, 0, 0, 0, 0, 0]
    for state_index, num_dice in enumerate(state):
        while num_dice > 0:
            keep = action[action_index]
            kept[state_index] += keep
            num_dice -= 1
            action_index += 1
            if action_index >= len(action):
                break

    return tuple(kept)


@lru_cache(maxsize=None)
def probability_of_goal_roll(goal_roll: tuple[int]) -> float:
    """
    Possibility of rolling sum(goal_roll) dice and
    getting the desired positive goal_roll values.

    For example, if goal_roll = [1, 1, 0, 0, 0, 0], then
    we are trying to roll 2 dice and get exactly one 1 and one 2.
    """
    # The number of dice we are trying to roll is just the sum of
    # the number of dices of each value we are trying to get.
    positive_goal_rolls = tuple(filter(lambda x: x > 0, goal_roll))
    if len(positive_goal_rolls) == 0:
        raise ValueError("Cannot roll 0 dice.")

    num_rolls = sum(positive_goal_rolls)

    # The total number of ways to roll num_rolls dice is 6^num_rolls.
    total_num_rolls = 6**num_rolls

    # The total number of ways to roll num_rolls dice and get exactly
    # the positive_goal_rolls values is the multinomial coefficient
    total_accepted_rolls = multinomial_coefficient(num_rolls, *positive_goal_rolls)

    return total_accepted_rolls / total_num_rolls


@lru_cache(maxsize=None)
def multinomial_coefficient(n: int, *k: int) -> int:
    """
    Calculate the multinomial coefficient (n choose k).
    """
    return round(factorial(n) / prod(factorial(k_i) for k_i in k))


def validate_transition_function(transition_function: TransitionFunctionType):
    """
    Validate the transition function.
    """
    for i in range(len(DICE_STATES)):
        for j in range(len(KEEP_ACTIONS)):
            # Check that each row sums to around 1 (floating point error)
            assert round(fsum(transition_function[i, j, :]), 15) == 1.0


def save_transition_function(transition_function: TransitionFunctionType):
    """
    Save the transition function to the TRANSITION_FUNCTION_FILE.
    """
    with open(TRANSITION_FUNCTION_FILE, "wb") as file:
        np.save(file, transition_function)


def load_transition_function() -> TransitionFunctionType:
    """
    Load the transition function from the TRANSITION_FUNCTION_FILE.
    """
    with open(TRANSITION_FUNCTION_FILE, "rb") as file:
        return np.load(file)


if __name__ == "__main__":
    print("Generating transition function...")
    result = generate_transition_function()
    print("Done.")
    print("Validating transition function...")
    validate_transition_function(result)
    print("Done.")
    print("Saving transition function...")
    save_transition_function(result)
    print("Done.")
