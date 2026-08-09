"""
SHARED FIGHTER CLASS —
BOTH THE HUMAN PLAYER AND THE AI ARE INSTANCES OF THIS CLASS,
DRIVEN EITHER BY REAL KEYBOARD INPUT OR BY SYNTHETIC "FORCED KEYS"
COMING FROM THE RL AGENT'S CHOSEN ACTION
"""

from collections import defaultdict

import pygame

import config

IDLE, RUN, JUMP, ATTACK1, ATTACK2, HIT, DEATH = range(7)

SPEED = 11
GRAVITY = 1.9
JUMP_VELOCITY = -27
ATTACK_RANGE = config.ATTACK_RANGE

LIGHT_ATTACK, HEAVY_ATTACK = 1, 2


class Fighter:
    def __init__(self, player, x, y, flip, sprite_sheet, frame_size, animation_steps, sound):
        self.player = player
        self.flip = flip
        self.animation_list, self.animation_offsets = self._load_images(
            sprite_sheet, frame_size, animation_steps
        )
        self.action = IDLE
        self.frame_index = 0
        self.image = self.animation_list[self.action][self.frame_index]
        self.update_time = pygame.time.get_ticks()

        self.rect = pygame.Rect(x, y, 80, 180)
        self.vel_y = 0
        self.running = False
        self.jump = False
        self.attacking = False
        self.attack_type = 0
        self.attack_cooldown = 0
        self.heavy_attack_cooldown = 0
        self.attack_sound = sound
        self.hit = False
        self.health = 100
        self.alive = True

    # -- setup --------------------------------------------------------
    @staticmethod
    def _load_images(sprite_sheet, size, animation_steps):
        scaled_size = int(size * config.SPRITE_SCALE)
        animation_list = []
        animation_offsets = []

        for row, frame_count in enumerate(animation_steps):
            frames = []
            offsets = []
            for col in range(frame_count):
                raw = sprite_sheet.subsurface(col * size, row * size, size, size)
                frame = pygame.transform.scale(raw, (scaled_size, scaled_size))
                frames.append(frame)
                offsets.append(Fighter._bottom_padding(frame))
            animation_list.append(frames)
            animation_offsets.append(offsets)

        return animation_list, animation_offsets

    @staticmethod
    def _bottom_padding(frame):
        mask = pygame.mask.from_surface(frame)
        rects = mask.get_bounding_rects()
        if not rects:
            return 0
        lowest_opaque_y = max(r.bottom for r in rects)
        return frame.get_height() - lowest_opaque_y

    # -- per-frame update ----------------------------------------------
    def move(self, screen_width, screen_height, target, round_over, forced_keys=None):
        dx = 0
        dy = 0
        self.running = False

        keys = forced_keys if forced_keys is not None else pygame.key.get_pressed()

        if not self.attacking and self.alive and not round_over:
            self.attack_type = 0
            left_key = pygame.K_a if self.player == 1 else pygame.K_LEFT
            right_key = pygame.K_d if self.player == 1 else pygame.K_RIGHT
            up_key = pygame.K_w if self.player == 1 else pygame.K_UP
            light_attack_key = pygame.K_r if self.player == 1 else pygame.K_p
            heavy_attack_key = pygame.K_t if self.player == 1 else pygame.K_o

            if keys[left_key]:
                dx = -SPEED
                self.running = True
            if keys[right_key]:
                dx = SPEED
                self.running = True
            if keys[up_key] and not self.jump:
                self.vel_y = JUMP_VELOCITY
                self.jump = True
            if keys[light_attack_key]:
                self.attack(target, LIGHT_ATTACK)
            elif keys[heavy_attack_key]:
                self.attack(target, HEAVY_ATTACK)

        self.vel_y += GRAVITY
        dy += self.vel_y

        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > screen_width:
            dx = screen_width - self.rect.right
        if self.rect.bottom + dy > screen_height - 110:
            self.vel_y = 0
            self.jump = False
            dy = screen_height - 110 - self.rect.bottom

        self.flip = target.rect.centerx < self.rect.centerx

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.heavy_attack_cooldown > 0:
            self.heavy_attack_cooldown -= 1

        self.rect.x += dx
        self.rect.y += dy

    @staticmethod
    def keys_from_action(action):
        """
        MAP A DISCRETE RL ACTION (0-5) TO A PYGAME-KEY-STYLE DICT
0       IDLE, 1 LEFT, 2 RIGHT, 3 JUMP, 4 LIGHT ATTACK, 5 HEAVY ATTACK
        """
        keys = defaultdict(bool)

        if action == 1:
            keys[pygame.K_a] = keys[pygame.K_LEFT] = True
        elif action == 2:
            keys[pygame.K_d] = keys[pygame.K_RIGHT] = True
        elif action == 3:
            keys[pygame.K_w] = keys[pygame.K_UP] = True
        elif action == 4:
            keys[pygame.K_r] = keys[pygame.K_p] = True
        elif action == 5:
            keys[pygame.K_t] = keys[pygame.K_o] = True

        return keys

    def update(self):
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self._set_action(DEATH)
        elif self.hit:
            self._set_action(HIT)
        elif self.attacking:
            self._set_action(ATTACK1 if self.attack_type == LIGHT_ATTACK else ATTACK2)
        elif self.jump:
            self._set_action(JUMP)
        elif self.running:
            self._set_action(RUN)
        else:
            self._set_action(IDLE)

        self.image = self.animation_list[self.action][self.frame_index]

        if pygame.time.get_ticks() - self.update_time > 50:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()

        if self.frame_index >= len(self.animation_list[self.action]):
            if not self.alive:
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:
                self.frame_index = 0
                if self.action in (ATTACK1, ATTACK2):
                    self._recover_from_attack()
                if self.action == HIT:
                    self.hit = False
                    self._recover_from_attack()

    def _recover_from_attack(self):
        self.attacking = False
        recovery = (
            config.LIGHT_ATTACK_RECOVERY_FRAMES
            if self.attack_type == LIGHT_ATTACK
            else config.HEAVY_ATTACK_RECOVERY_FRAMES
        )
        self.attack_cooldown = recovery

    def attack(self, target, attack_type=LIGHT_ATTACK):
        if self.attack_cooldown != 0:
            return
        if attack_type == HEAVY_ATTACK and self.heavy_attack_cooldown != 0:
            return

        self.attacking = True
        self.attack_type = attack_type
        self.attack_sound.play()

        if attack_type == HEAVY_ATTACK:
            self.heavy_attack_cooldown = config.HEAVY_ATTACK_COOLDOWN_FRAMES

        reach = pygame.Rect(
            self.rect.centerx - (ATTACK_RANGE * self.flip),
            self.rect.y, ATTACK_RANGE, self.rect.height
        )
        if reach.colliderect(target.rect):
            damage = (
                config.LIGHT_ATTACK_DAMAGE
                if attack_type == LIGHT_ATTACK
                else config.HEAVY_ATTACK_DAMAGE
            )
            target.health -= damage
            target.hit = True

    def _set_action(self, new_action):
        if new_action != self.action:
            self.action = new_action
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def reset(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.health = 100
        self.alive = True
        self.vel_y = 0
        self.jump = False
        self.attacking = False
        self.attack_cooldown = 0
        self.heavy_attack_cooldown = 0
        self.hit = False
        self.frame_index = 0
        self.action = IDLE

    def draw(self, surface):
        img = pygame.transform.flip(self.image, self.flip, False)
        padding = self.animation_offsets[self.action][self.frame_index]
        x = self.rect.centerx - img.get_width() // 2
        y = self.rect.bottom - img.get_height() + padding
        surface.blit(img, (x, y))
