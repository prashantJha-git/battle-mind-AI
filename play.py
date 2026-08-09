"""RUN THIS FILE TO PLAY"""

import os
import sys

import config

REQUIRED_ASSETS = [
    config.BACKGROUND_IMAGE, config.PLAYER_SPRITE, config.AI_SPRITE,
    config.ATTACK_SFX, config.MUSIC_FILE,
]


def check_assets():
    missing = [path for path in REQUIRED_ASSETS if not os.path.exists(path)]
    if missing:
        print("Missing asset files:")
        for path in missing:
            print(f"  - {path}")
        print("\nMake sure the assets/ folder is next to this script.")
        sys.exit(1)


if __name__ == "__main__":
    check_assets()
    import game
    game.main()
