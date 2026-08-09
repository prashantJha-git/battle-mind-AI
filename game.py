import sys
import random
from collections import deque

import numpy as np
import pygame
from pygame import mixer

import config
from fighter import Fighter, LIGHT_ATTACK, HEAVY_ATTACK
from rlAgent import RLAgent
from stateExtractor import StateExtractor
from reactionTracker import ReactionTracker

ATTACK_FLASH_MS = 550
HEALTH_SEGMENTS = 10

DODGE_DISTANCE = 150

EXPLORATION_SCHEDULE = [
    (5, 0.30),
    (10, 0.18),
    (20, 0.10),
]


class Game:
    def __init__(self):
        pygame.init()
        mixer.init()

        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("BATTLE-MIND AI")
        self.clock = pygame.time.Clock()

        self._load_assets()

        self.agent = RLAgent(config.STATE_SIZE, config.ACTION_SIZE, config.LEARNING_RATE)
        try:
            self.agent.load_weights(config.MODEL_FILE)
            print(f"\n[LOADING] >>> LOADED EXISTING AI MODEL FROM :: {config.MODEL_FILE} ... ")
        except FileNotFoundError:
            print("\n[LOADING] >>> NO SAVED MODEL FOUND :: AI STARTS UNTRAINED ... LEARNS AS YOU PLAY ... ")

        self.state_extractor = StateExtractor(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.reaction = ReactionTracker()

        self.score = [0, 0]
        self.results = deque(maxlen=10)
        self.round_over = False
        self.round_over_time = 0
        self.round_winner = None  # "player" | "ai" | None
        self.intro_count = 3
        self.last_count_update = pygame.time.get_ticks()

        self.rounds_played = 1
        self.episode_step = 0
        self.ai_action_counter = 0
        self.ai_current_action = 0

        self.player = Fighter(1, 200, 310, False, self.player_sheet,
                              config.PLAYER_FRAME_SIZE, config.PLAYER_ANIMATION_STEPS,
                              self.attack_fx)
        self.ai = Fighter(2, 700, 310, True, self.ai_sheet,
                          config.AI_FRAME_SIZE, config.AI_ANIMATION_STEPS,
                          self.attack_fx)

        self.displayed_health = [100.0, 100.0]

        self.attack_flash = [{"type": None, "time": 0}, {"type": None, "time": 0}]
        self._prev_attacking = [False, False]

    # -- setup -------------------------------------------------------------
    def _load_assets(self):
        self.bg_image = pygame.image.load(config.BACKGROUND_IMAGE).convert()
        self.player_sheet = pygame.image.load(config.PLAYER_SPRITE).convert_alpha()
        self.ai_sheet = pygame.image.load(config.AI_SPRITE).convert_alpha()

        self.attack_fx = mixer.Sound(config.ATTACK_SFX)
        self.attack_fx.set_volume(0.6)

        mixer.music.load(config.MUSIC_FILE)
        mixer.music.set_volume(0.4)
        mixer.music.play(-1)

        self.count_font = self._load_font(100, bold=True)
        self.title_font = self._load_font(46, bold=True)
        self.name_font = self._load_font(30, bold=True)
        self.round_font = self._load_font(24, bold=True)
        self.indicator_font = self._load_font(15, bold=True)
        self.small_font = self._load_font(16, bold=True)

    @staticmethod
    def _load_font(size, bold=False):
        for name in config.FONT_CANDIDATES:
            if pygame.font.match_font(name, bold=bold):
                return pygame.font.SysFont(name, size, bold=bold)
        font = pygame.font.Font(None, size)
        font.set_bold(bold)
        return font

    # -- small drawing helpers ----------------------------------------------
    def _draw_outlined_text(self, text, font, color, x, y, outline=config.BLACK,
                            outline_width=2, center=True):
        pos = (x, y)
        offsets = [
            (-outline_width, 0), (outline_width, 0), (0, -outline_width), (0, outline_width),
            (-outline_width, -outline_width), (outline_width, -outline_width),
            (-outline_width, outline_width), (outline_width, outline_width),
        ]
        outline_surf = font.render(text, True, outline)
        for dx, dy in offsets:
            opos = (x + dx, y + dy)
            rect = outline_surf.get_rect(center=opos) if center else outline_surf.get_rect(topleft=opos)
            self.screen.blit(outline_surf, rect)

        surf = font.render(text, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def _draw_bg(self):
        self.screen.blit(self.bg_image, (0, 0))

    def _dim_screen(self, alpha):
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((*config.BLACK, alpha))
        self.screen.blit(overlay, (0, 0))

    # -- health bar ------------------------------------------------------
    def _health_color(self, ratio):
        if ratio > 0.5:
            t = (ratio - 0.5) / 0.5
            return self._lerp_color(config.HP_MID, config.HP_HIGH, t)
        t = ratio / 0.5
        return self._lerp_color(config.HP_LOW, config.HP_MID, t)

    @staticmethod
    def _lerp_color(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _draw_segmented_health_bar(self, displayed_health, x, y, width, height=26,
                                   segments=HEALTH_SEGMENTS, flip=False):
        skew = height // 3
        gap = 3
        seg_width = (width - gap * (segments - 1)) / segments
        seg_health = 100 / segments
        color = self._health_color(max(displayed_health, 0) / 100)

        for i in range(segments):
            seg_x = x + i * (seg_width + gap)
            points = [
                (seg_x + skew, y),
                (seg_x + seg_width, y),
                (seg_x + seg_width - skew, y + height),
                (seg_x, y + height),
            ]

            tier = i if not flip else segments - 1 - i
            tier_start = tier * seg_health
            fill_ratio = min(1.0, max(0.0, (displayed_health - tier_start) / seg_health))

            pygame.draw.polygon(self.screen, config.HP_TRACK, points)

            if fill_ratio > 0:
                clip_w = seg_width * fill_ratio
                clip_x = seg_x if not flip else seg_x + seg_width - clip_w
                clip_rect = pygame.Rect(int(clip_x) - 1, y - 2, int(clip_w) + 2, height + 4)
                prev_clip = self.screen.get_clip()
                self.screen.set_clip(clip_rect)
                pygame.draw.polygon(self.screen, color, points)
                self.screen.set_clip(prev_clip)

            pygame.draw.polygon(self.screen, config.HP_SEGMENT_BORDER, points, width=2)

    # -- attack indicator --------------------------------------------------
    def _update_attack_flashes(self):
        for i, fighter in enumerate((self.player, self.ai)):
            just_started = fighter.attacking and not self._prev_attacking[i]
            if just_started:
                self.attack_flash[i] = {"type": fighter.attack_type, "time": pygame.time.get_ticks()}
            self._prev_attacking[i] = fighter.attacking

    def _draw_attack_indicator(self, x, y, flash, align_right=False):
        now = pygame.time.get_ticks()
        active = flash["type"] if now - flash["time"] < ATTACK_FLASH_MS else None

        labels = [("FAST", LIGHT_ATTACK, config.ACCENT), ("HEAVY", HEAVY_ATTACK, config.ACCENT_WARM)]
        if align_right:
            labels = list(reversed(labels))

        cursor_x = x
        for text, atype, color in labels:
            is_active = atype == active
            draw_color = color if is_active else config.MUTED_TEXT
            outline = config.BLACK if is_active else (40, 40, 46)
            width, _ = self.indicator_font.size(text)

            pos_x = cursor_x + width // 2 if not align_right else cursor_x - width // 2
            self._draw_outlined_text(text, self.indicator_font, draw_color, pos_x, y,
                                     outline=outline, outline_width=2 if is_active else 1)

            if not align_right:
                cursor_x += width + 14
            else:
                cursor_x -= width + 14

    # -- HUD -----------------------------------------------------------------
    def _draw_hud(self):
        bar_width = 380
        margin = 26
        left_x, right_x = margin, config.SCREEN_WIDTH - margin - bar_width
        bar_y = 26

        self._draw_segmented_health_bar(self.displayed_health[0], left_x, bar_y, bar_width, flip=False)
        self._draw_segmented_health_bar(self.displayed_health[1], right_x, bar_y, bar_width, flip=True)

        self._draw_attack_indicator(left_x, bar_y + 44, self.attack_flash[0], align_right=False)
        self._draw_attack_indicator(right_x + bar_width, bar_y + 44, self.attack_flash[1], align_right=True)

        self._draw_outlined_text("YOU", self.name_font, config.NAME_GOLD,
                                 left_x, bar_y + 74, center=False)
        ai_w, _ = self.name_font.size("AI")
        self._draw_outlined_text("AI", self.name_font, config.NAME_GOLD,
                                 right_x + bar_width - ai_w, bar_y + 74, center=False)

        self._draw_outlined_text(f"WINS {self.score[0]}", self.small_font, config.WHITE,
                                 left_x, bar_y + 104, center=False)
        wins_text = f"WINS {self.score[1]}"
        wins_w, _ = self.small_font.size(wins_text)
        self._draw_outlined_text(wins_text, self.small_font, config.WHITE,
                                 right_x + bar_width - wins_w, bar_y + 104, center=False)

        center_x = config.SCREEN_WIDTH // 2
        self._draw_outlined_text(f"ROUND {self.rounds_played}", self.round_font,
                                 config.WHITE, center_x, bar_y + 14)
        win_rate = (sum(self.results) / len(self.results) * 100) if self.results else 0
        self._draw_outlined_text(f"AI WIN RATE {win_rate:.0f}%", self.small_font,
                                 config.YELLOW, center_x, bar_y + 40, outline_width=1)

    def _update_displayed_health(self):
        speed = config.HEALTH_BAR_LERP_SPEED
        self.displayed_health[0] += (self.player.health - self.displayed_health[0]) * speed
        self.displayed_health[1] += (self.ai.health - self.displayed_health[1]) * speed
        if abs(self.displayed_health[0] - self.player.health) < 0.5:
            self.displayed_health[0] = self.player.health
        if abs(self.displayed_health[1] - self.ai.health) < 0.5:
            self.displayed_health[1] = self.ai.health

    # -- round management ------------------------------------------------
    def _reset_round(self):
        self.round_over = False
        self.round_winner = None
        self.intro_count = 3
        self.last_count_update = pygame.time.get_ticks()
        self.episode_step = 0
        self.ai_action_counter = 0
        self.ai_current_action = 0
        self.player.reset(200, 310)
        self.ai.reset(700, 310)
        self.displayed_health = [100.0, 100.0]
        self.attack_flash = [{"type": None, "time": 0}, {"type": None, "time": 0}]
        self._prev_attacking = [False, False]

    def _reward(self, prev_ai_health, prev_player_health, prev_distance, done):
        reward = 0.0

        damage_done = prev_player_health - self.player.health
        reward += damage_done * config.DAMAGE_DEALT_REWARD

        damage_taken = prev_ai_health - self.ai.health
        reward -= damage_taken * config.DAMAGE_TAKEN_PENALTY

        current_distance = abs(self.ai.rect.centerx - self.player.rect.centerx)
        if current_distance < prev_distance:
            reward += config.MOVE_TOWARDS_REWARD
        else:
            reward -= config.MOVE_AWAY_PENALTY

        if current_distance < 140:
            reward += 0.12

        if self.ai.attacking:
            is_heavy = self.ai.attack_type == HEAVY_ATTACK
            if damage_done > 0:
                reward += config.SUCCESSFUL_HEAVY_ATTACK_REWARD if is_heavy else config.SUCCESSFUL_ATTACK_REWARD
            else:
                reward -= config.MISSED_HEAVY_ATTACK_PENALTY if is_heavy else config.MISSED_ATTACK_PENALTY

        if self.player.attacking and damage_taken == 0:
            reward += 0.5

        if self.ai_current_action == 0:
            reward -= config.IDLE_PENALTY

        if self.ai_current_action == 3 and current_distance > 220:
            reward -= 0.10

        if done:
            if self.player.health <= 0:
                reward += config.ROUND_WIN_BONUS
            elif self.ai.health <= 0:
                reward -= config.ROUND_LOSS_PENALTY

        return reward

    # -- main loop --------------------------------------------------------
    def run(self):
        running = True
        while running:
            self.clock.tick(config.FPS)
            self.reaction.tick()

            self._draw_bg()

            if self.intro_count > 0:
                self._draw_intro_countdown()
                if pygame.time.get_ticks() - self.last_count_update >= 1000:
                    self.intro_count -= 1
                    self.last_count_update = pygame.time.get_ticks()
            else:
                self.episode_step += 1
                self._step_player()
                self._step_ai()

            self.player.update()
            self.ai.update()
            self.player.draw(self.screen)
            self.ai.draw(self.screen)

            self._update_attack_flashes()
            self._update_displayed_health()
            self._draw_hud()

            self._check_round_end()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            pygame.display.update()

        self._shutdown()

    def _draw_intro_countdown(self):
        self._dim_screen(90)
        cx, cy = config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 3
        self._draw_outlined_text(str(self.intro_count), self.count_font, config.YELLOW,
                                 cx, cy, outline_width=4)

    def _step_player(self):
        self.player.move(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, self.ai, self.round_over)
        is_acting = self.player.running or self.player.jump or self.player.attacking
        self.reaction.record_player_state(is_acting)

    def _current_exploration_target(self):
        for rounds_threshold, exploration in EXPLORATION_SCHEDULE:
            if self.rounds_played < rounds_threshold:
                return exploration
        return config.EXPLORATION_MIN

    def _step_ai(self):
        delay = self.reaction.current_delay()
        self.ai_action_counter += 1

        if self.ai_action_counter >= delay:
            self.ai_action_counter = 0

            state = self.state_extractor.extract_state(self.ai, self.player)
            prev_ai_health = self.ai.health
            prev_player_health = self.player.health
            prev_distance = abs(self.ai.rect.centerx - self.player.rect.centerx)

            self.agent.exploration = self._current_exploration_target()

            action = self._choose_ai_action(state, prev_distance)
            self.ai_current_action = action

            keys = Fighter.keys_from_action(action)
            self.ai.move(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, self.player, self.round_over, forced_keys=keys)

            done = not (self.ai.alive and self.player.alive)
            reward = self._reward(prev_ai_health, prev_player_health, prev_distance, done)
            self.agent.store_experience(state, action, reward, done)

            if self.agent.buffer_size() >= config.TRAIN_EVERY_STEPS:
                self.agent.train()
        else:
            keys = Fighter.keys_from_action(self.ai_current_action)
            self.ai.move(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, self.player, self.round_over, forced_keys=keys)

    def _choose_ai_action(self, state, distance):
        """
        BLEND SCRIPTED POSITIONING (PUNISH / CHASE / ATTACK RANGE)
        WITH THE LEARNED POLICY FOR EVERYTHING IN BETWEEN
        """
        heavy_ready = self.ai.attack_cooldown == 0 and self.ai.heavy_attack_cooldown == 0

        if self.player.attacking and distance < DODGE_DISTANCE:
            if heavy_ready and random.random() < 0.4:
                return 5  # counter-attack with a heavy
            return random.choice([1, 2, 3])  # dodge left / right / jump

        if distance > 190:
            return 2 if self.ai.rect.centerx < self.player.rect.centerx else 1  # chase

        if distance < 150:
            if not heavy_ready:
                return 4
            if self.player.health <= 30:
                return int(np.random.choice([4, 5], p=[0.35, 0.65]))
            return int(np.random.choice([4, 5], p=[0.65, 0.35]))

        return self.agent.get_action(state)

    def _check_round_end(self):
        if not self.round_over:
            if not self.player.alive:
                self.score[1] += 1
                self.results.append(1)
                self.round_winner = "ai"
                self._end_round()
            elif not self.ai.alive:
                self.score[0] += 1
                self.results.append(0)
                self.round_winner = "player"
                self._end_round()
            elif self.episode_step > config.MAX_STEPS_PER_ROUND:
                self._end_round()
        else:
            self._draw_round_over_banner()
            if pygame.time.get_ticks() - self.round_over_time > 1200:
                self.rounds_played += 1
                self.agent.train()
                self.agent.save_weights(config.MODEL_FILE)
                self._reset_round()

    def _draw_round_over_banner(self):
        self._dim_screen(110)
        cx, cy = config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 3
        self._draw_outlined_text("ROUND OVER", self.title_font, config.WHITE, cx, cy, outline_width=3)

        subtitle = {
            "player": ("YOU WIN THE ROUND", config.ACCENT),
            "ai": ("AI WINS THE ROUND", config.ACCENT_WARM),
        }.get(self.round_winner, ("TIME UP", config.MUTED_TEXT))
        self._draw_outlined_text(subtitle[0], self.round_font, subtitle[1], cx, cy + 40)

    def _end_round(self):
        self.round_over = True
        self.round_over_time = pygame.time.get_ticks()

    def _shutdown(self):
        print(f"\n[BATTLE LOG]\n\n[FINAL_SCORE] >>> YOU:{self.score[0]} :: AI:{self.score[1]}")
        print(f"ROUNDS PLAYED: {self.rounds_played}")
        self.agent.train()
        self.agent.save_weights(config.MODEL_FILE)
        print(f"\n\n[TRAINING ARCHIVE]\n[SAVING] >>> MODEL SAVED TO:{config.MODEL_FILE}")
        pygame.quit()
        sys.exit()


def main():
    Game().run()


if __name__ == "__main__":
    main()
