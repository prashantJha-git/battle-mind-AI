from collections import deque
import config


class ReactionTracker:
    """
    ESTIMATES HOW QUICKLY THE PLAYER MAKES DECISIONS AND
    FEEDS THAT ESTIMATE TO THE AI AS ITS OWN DECISION DELAY,
    SO IT NEITHER STALLS NOR REACTS INHUMANLY FAST
    """

    def __init__(self):
        self.frame_count = 0
        self.last_action_frame = 0
        self.gaps = deque(maxlen=config.REACTION_SAMPLE_WINDOW)
        self.ai_delay = float(config.AI_DELAY_DEFAULT)
        self._prev_active = False

    def tick(self):
        self.frame_count += 1

    def record_player_state(self, is_acting):
        """
        CALL ONCE PER FRAME WITH WHETHER THE PLAYER IS CURRENTLY
        MOVING, JUMPING, OR ATTACKING.
        A NEW ACTION IS COUNTED ON THE RISING EDGE
        """
        if is_acting and not self._prev_active:
            gap = self.frame_count - self.last_action_frame
            if 0 < gap < config.AI_DELAY_MAX * 6:
                self.gaps.append(gap)
            self.last_action_frame = self.frame_count
        self._prev_active = is_acting

    def current_delay(self):
        if self.gaps:
            target = sum(self.gaps) / len(self.gaps)
            target = max(config.AI_DELAY_MIN, min(config.AI_DELAY_MAX, target))
            self.ai_delay = (config.REACTION_SMOOTHING * self.ai_delay
                             + (1 - config.REACTION_SMOOTHING) * target)
        return max(config.AI_DELAY_MIN, min(config.AI_DELAY_MAX, round(self.ai_delay)))
