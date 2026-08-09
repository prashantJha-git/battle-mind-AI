import numpy as np


class StateExtractor:
    """TURNS TWO FIGHTER INSTANCES INTO THE 12-VALUE STATE VECTOR THE RL AGENT SEES"""

    ACTIONS = {
        0: "idle",
        1: "move_left",
        2: "move_right",
        3: "jump",
        4: "light_attack",
        5: "heavy_attack",
    }

    def __init__(self, screen_width=1000, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def extract_state(self, fighter_self, fighter_opponent):
        rel_x = (fighter_opponent.rect.centerx - fighter_self.rect.centerx) / self.screen_width
        rel_y = (fighter_opponent.rect.centery - fighter_self.rect.centery) / self.screen_height

        self_health = fighter_self.health / 100.0
        opp_health = fighter_opponent.health / 100.0

        self_vel_y = np.tanh(fighter_self.vel_y / 30.0)
        opp_vel_y = np.tanh(fighter_opponent.vel_y / 30.0)
        self_vel_x = np.tanh(fighter_self.rect.x / self.screen_width)
        opp_vel_x = np.tanh(fighter_opponent.rect.x / self.screen_width)

        opponent_attacking = 1.0 if fighter_opponent.attacking else 0.0
        opponent_heavy_attack = 1.0 if fighter_opponent.attack_type == 2 else 0.0
        self_attacking = 1.0 if fighter_self.attacking else 0.0

        self_grounded = 1.0 if not fighter_self.jump else 0.0
        opponent_grounded = 1.0 if not fighter_opponent.jump else 0.0

        return np.array(
            [
                rel_x, rel_y,
                self_health, opp_health,
                self_vel_x, self_vel_y,
                opp_vel_x, opp_vel_y,
                opponent_attacking, opponent_heavy_attack,
                self_grounded, opponent_grounded,
            ],
            dtype=np.float32
        )
