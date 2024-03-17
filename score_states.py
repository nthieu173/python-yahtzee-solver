from enum import Enum

from dice_states import DiceStateType


class ScoreAction(Enum):
    """
    Which score category to apply the dice state to.
    """

    ONES = 0
    TWOS = 1
    THREES = 2
    FOURS = 3
    FIVES = 4
    SIXES = 5

    THREE_OF_A_KIND = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    CHANCE = 11

    YAHTZEE = 12


InnerScoreState = tuple[
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool | None,
]


class ScoreState:
    """
    A score state of a player in a single round of Yahtzee.
    """

    NUM_STATES: int = 2**26 - 2
    NUM_TERMINAL_STATES = 7**6 * 2

    def __init__(self, state: InnerScoreState):
        self.state = state
        self._as_int_cache = None

    def __repr__(self) -> str:
        return f"ScoreState{self.state}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreState):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        return self.as_int()

    def __copy__(self) -> "ScoreState":
        return ScoreState(self.state)

    def as_int(self) -> int:
        """
        Return the state as an int from 0 to 2**26 - 2.
        """
        if self._as_int_cache is not None:
            return self._as_int_cache
        bits = 0
        # Represent the first 6 values as 3 bits. None is 7.
        for index, value in enumerate(self.state[:6]):
            if value is None:
                value = 7
            bits |= value << (index * 3)
        # Length so far: 18
        # The next 6 values are 1 bit. False is 0.
        for index, value in enumerate(self.state[6:12], 18):
            bits |= int(value) << index
        # Length so far: 24
        # The last value is 2 bits. None is 2. Thus, we will only waste 1 bit.
        value = self.state[12]
        if value is None:
            value = 2
        bits |= value << 24
        # Length so far: 26
        return bits

    @classmethod
    def from_int(cls, state_int: int):
        """
        Set the state from an int. The int must be from 0 to 2**26 - 2.
        """
        if 0 > state_int or state_int > cls.NUM_STATES:
            raise ValueError("Invalid state int")
        bits = f"{state_int:026b}"
        state = [None] * 13
        for index in range(6):
            shift = index * 3
            mask = 7 << shift
            value = (state_int & mask) >> shift
            if value == 7:
                value = None
            state[index] = value
        for index in range(6, 12):
            shift = index
            mask = 1 << shift
            value = (state_int & mask) >> shift
            state[index] = bool(value)
        shift = 12
        mask = 3 << shift
        value = (state_int & mask) >> shift
        state[12] = bool(value) if value != 2 else None
        return cls(tuple(state))

    def apply_action(
        self, action: ScoreAction, dice_state: DiceStateType
    ) -> "ScoreState":
        """
        Apply the given action to this state. Return the new state.
        """
        if action not in ScoreAction:
            raise ValueError(f"Invalid action: {action}")
        action_index = action.value
        new_state = list(self.state)
        if action_index < 6:
            new_state[action_index] = dice_state[action_index]
        elif action_index < 12:
            new_state[action_index] = True
        else:
            new_state[action_index] = self.is_yahtzee(dice_state)
        return ScoreState(tuple(new_state))

    def parent_states(self) -> list["ScoreState"]:
        """
        Return the parent states and actions that lead to this state.
        """
        parent_states = []
        # Simple values
        for index in range(6):
            if self.state[index] is not None:
                new_state = list(self.state)
                new_state[index] = None
                parent_state = ScoreState(tuple(new_state))
                parent_states.append(parent_state)
        for value in range(6, 12):
            if self.state[value]:
                new_state = list(self.state)
                new_state[value] = False
                parent_state = ScoreState(tuple(new_state))
                parent_states.append(parent_state)
        if self.state[12] is not None:
            new_state = list(self.state)
            new_state[12] = None
            parent_state = ScoreState(tuple(new_state))
            parent_states.append(parent_state)
        return parent_states

    def possible_actions(self) -> list[ScoreAction]:
        """
        Return the actions that can be taken from this state.
        """
        actions = []
        # Simple values
        for index in range(7):
            if self.state[index] is None:
                actions.append(ScoreAction(index))
        # Boolean values
        for value in range(6, 12):
            if not self.state[value]:
                actions.append(ScoreAction(value))
        if self.state[12] is None:
            actions.append(ScoreAction(12))
        return actions

    def is_terminal(self) -> bool:
        return len(self.possible_actions()) == 0

    def reward(self, score_action: ScoreAction, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for the given score action.
        """
        action_index = score_action.value
        if action_index < 6:
            return self.simple_reward(dice_state, action_index + 1)
        if action_index == 6:
            return self.three_of_a_kind_reward(dice_state)
        if action_index == 7:
            return self.four_of_a_kind_reward(dice_state)
        if action_index == 8:
            return self.full_house_reward(dice_state)
        if action_index == 9:
            return self.small_straight_reward(dice_state)
        if action_index == 10:
            return self.large_straight_reward(dice_state)
        if action_index == 11:
            return self.chance_reward(dice_state)
        if action_index == 12:
            return self.yahtzee_reward(dice_state)

        raise ValueError(f"Invalid action: {score_action}")

    def simple_reward(self, dice_state: DiceStateType, value: int) -> int:
        """
        Return the reward for using the give dice state for the given simple value.
        """
        value_index = value - 1
        if value_index < 0 or value_index >= 6:
            raise ValueError(f"Invalid dice value: {value}")
        num_dice = dice_state[value_index]
        if num_dice == 0:
            return 0

        simples = list(self.state[:6])
        before_sum_of_simples = self.sum_of_simples(simples)

        simples[value_index] = num_dice
        after_sum_of_simples = self.sum_of_simples(simples)

        reward = num_dice * value
        if before_sum_of_simples < 63 <= after_sum_of_simples:
            # Bonus for getting 63 or more in simples
            reward += 35
        return reward

    def three_of_a_kind_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for three of a kind.
        """
        for value in range(1, 7):
            if dice_state[value - 1] >= 3:
                reward = 0
                if dice_state[value - 1] == 5 and self.state[12]:
                    # If there are any Yahtzees, we add 100 if the Yahtzee is already taken.
                    reward += 100
                # If there are any three of a kind,
                # then the reward is the sum of all dices.
                return reward + sum(
                    val * count for val, count in enumerate(dice_state, start=1)
                )
        return 0

    def four_of_a_kind_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for four of a kind.
        """
        for value in range(1, 7):
            if dice_state[value - 1] >= 4:
                reward = 0
                if dice_state[value - 1] == 5 and self.state[12]:
                    # If there are any Yahtzees, we add 100 if the Yahtzee is already taken.
                    reward += 100
                # If there are any four of a kind,
                # then the reward is the sum of all dices.
                return reward + sum(
                    val * count for val, count in enumerate(dice_state, start=1)
                )
        return 0

    def full_house_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for a full house.
        """
        reward = 25
        # A Yahztee is also a full house, so we need to check for that first.
        if self.is_yahtzee(dice_state):
            # If the Yahtzee is already taken, then the reward has an additional 100.
            if self.state[12]:
                reward += 100
            return reward

        for triple_value in range(1, 7):
            for pair_value in range(1, 7):
                if (
                    triple_value != pair_value
                    and dice_state[triple_value - 1] == 3
                    and dice_state[pair_value - 1] == 2
                ):
                    # If there are any full houses,
                    # then the reward is 25.
                    return reward
        return 0

    def small_straight_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for a small straight.
        """
        reward = 30
        # A Yahztee is also a small straight, so we need to check for that first.
        if self.is_yahtzee(dice_state):
            if self.state[12]:
                # If the Yahtzee is already taken, then the reward has an additional 100.
                reward += 100
            return reward

        for value in range(1, 4):
            # 3 possible small straights (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)
            if all(dice_state[value - 1 : value + 4]):
                # If there are any small straights,
                # then the reward is 30.
                return 30
        return 0

    def large_straight_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for a large straight.
        """
        reward = 40
        # A Yahztee is also a large straight, so we need to check for that first.
        if self.is_yahtzee(dice_state):
            if self.state[12]:
                # If the Yahtzee is already taken, then the reward has an additional 100.
                reward += 100
            return reward

        for value in range(1, 3):
            # 2 possible large straights (1, 2, 3, 4, 5), (2, 3, 4, 5, 6)
            if all(dice_state[value - 1 : value + 5]):
                # If there are any large straights,
                # then the reward is 40.
                return 40
        return 0

    def chance_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for a chance.
        """
        reward = sum(val * count for val, count in enumerate(dice_state, start=1))
        if self.is_yahtzee(dice_state) and self.state[12]:
            # If there are any Yahtzees, we add 100 if the Yahtzee is already taken.
            reward += 100
        return reward

    def yahtzee_reward(self, dice_state: DiceStateType) -> int:
        """
        Return the reward for using the given dice state for a yahtzee.
        """
        if self.is_yahtzee(dice_state):
            # If there are any Yahtzees,
            # then the reward is 50.
            return 50
        return 0

    @classmethod
    def is_yahtzee(cls, dice_state: DiceStateType) -> bool:
        """
        Return True if the given dice state is a Yahtzee.
        """
        return any(map(lambda x: x == 5, dice_state))

    @classmethod
    def sum_of_simples(cls, simple_state: list[int | None]) -> int | None:
        """
        Return the sum of the simple scores. None are treated as 0.
        """
        return sum(
            value * (0 if count is None else count)
            for value, count in enumerate(simple_state, start=1)
        )

    @classmethod
    def get_all_terminal_state(cls) -> list["ScoreState"]:
        """
        Return all possible terminal states of Yahtzee.
        """
        terminal_states = []
        for (
            num_ones,
            num_twos,
            num_threes,
            num_fours,
            num_fives,
            num_sixes,
            yahtzee_gotten,
        ) in [
            (
                num_ones,
                num_twos,
                num_threes,
                num_fours,
                num_fives,
                num_sixes,
                yahtzee_gotten,
            )
            for num_ones in range(7)
            for num_twos in range(7)
            for num_threes in range(7)
            for num_fours in range(7)
            for num_fives in range(7)
            for num_sixes in range(7)
            for yahtzee_gotten in (False, True)
        ]:
            state = (
                # Must be taken with a value from 0 to 6
                num_ones,
                num_twos,
                num_threes,
                num_fours,
                num_fives,
                num_sixes,
                # Must be taken
                True,
                True,
                True,
                True,
                True,
                True,
                # True or False depending on whether it was taken or 0
                yahtzee_gotten,
            )
            score_state = ScoreState(state)
            terminal_states.append(score_state)

        return terminal_states
