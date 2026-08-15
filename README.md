```
# BATTLE-MIND AI
```

A 2D fighter game where the opponent is a policy-gradient reinforcement learning agent that trains continuously in real time.

The goal of this project is to practice key game-dev and machine-learning concepts, including:

- Sprite-based animation and hit detection
- State representation for a reinforcement-learning agent
- Policy-gradient training (REINFORCE + value baseline) written from scratch in NumPy
- Reward shaping for combat behavior
- Adaptive AI pacing based on player reaction time
- Persisting and reloading trained weights across sessions

---
```
# PLAY NOW
```

Want To jump straight in without setting up Python or any dependencies? Download the playable build from itch.io, extract the zip, and run it:

https://aahan-chaturvedi.itch.io/battlemind

This deployed, playable build was put together by [Aahan Chaturvedi](https://github.com/Aahan-Chaturvedi), who worked on this project as a collaborator.

If you'd rather run the game from source (e.g. to tweak the RL agent, reward shaping, or bring your own assets), follow the "How to Run" instructions further down this page instead.

---

```
# Project Description
```

The AI opponent is trained live, during play — there is no separate offline training phase.

The main pieces of the project are:

| File | Responsibility |
|:--|:--|
| `fighter.py` | Shared player/AI character class — animation, movement, attacks |
| `stateExtractor.py` | Converts fighter state into the agent's 12-value input vector |
| `rlAgent.py` | Policy-gradient RL agent — chooses actions, trains on stored experience |
| `neuralNetwork.py` | Minimal NumPy feedforward network (no ML framework deps) |
| `reactionTracker.py` | Adapts the AI's reaction delay to the player's own pace |
| `game.py` | Main loop, HUD rendering, round logic, reward shaping |

Agent weights are saved to `model.npy` after every round and reloaded on startup, so the AI keeps improving across sessions.

This project was developed as part of a team's learning process in game development and reinforcement learning.

---

```
# Model Information
```

Algorithm: REINFORCE with a value-function baseline  
Architecture: Policy network (2x64 hidden, ReLU) + Value network (64 hidden, ReLU)  
State Size: 12  
Action Size: 6 (idle, left, right, jump, light attack, heavy attack)  
Learning Rate: 0.01 (policy), 0.001 (value)  
Exploration: starts at 0.35, decays to a 0.02 floor  
Training Cadence: every 20 stored transitions, and once at the end of each round

---

```
# Technologies Used
```

- PyGame
- NumPy

---

```
# Project Structure
```

```
BATTLE-MIND-AI/
│
├── game.py
├── fighter.py
├── rlAgent.py
├── neuralNetwork.py
├── stateExtractor.py
├── reactionTracker.py
├── config.py
├── play.py
├── requirements.txt
├── README.md
├── LICENSE
├── LICENSE-background.txt
├── .gitignore
└── assets/   (sprites, background, audio — not included)
```

---

```
# Assets
```

Sprites, background art, and audio are not included in this repository because of file size and licensing.

To run this project:

1. Add your own assets under the `assets/` folder, matching the paths defined in `config.py`:

```
assets/images/player/sprites/player.png
assets/images/ai/sprites/ai.png
assets/images/background/background.png
assets/audio/attack.wav
assets/audio/music.mp3
```

2. `play.py` checks that every required asset exists before launching and tells you exactly what's missing if something isn't there.

---

```
# How to Run
```

Clone the repository:

```
git clone https://github.com/prashantJha-git/battle-mind-AI.git
cd battle-mind-AI
```

Install required libraries:

```
pip install -r requirements.txt
```

Run the game:

```
python play.py
```

---

```
# Controls
```

| Action | Player 1 | AI (internal) |
|:--|:--:|:--:|
| Move | A / D | ← / → |
| Jump | W | ↑ |
| Light attack | R | P |
| Heavy attack | T | O |

Light attacks are fast and recover quickly; heavy attacks hit harder but leave you open longer and sit on a 1.5s cooldown (90 frames at 60 FPS), so they read as a real hard-hitting option rather than something to spam.

---

```
# AI Model Persistence
```

Weights are saved to `model.npy` after every round and reloaded on startup, so progress persists across sessions. `model.npy` is git-ignored — it's a generated artifact, not source.

---

```
# License
```

Code is licensed under the MIT License.

Background art is used under the terms in `LICENSE-background.txt`:

You can use any asset personally and commercially. You can modify/adapt the asset. You must give appropriate credit. You can NOT re-distribute the files, no matter how much you modify it — you can use it but not share or re-sell it as an asset.

Credit: add the background artist's name/handle/link here.

Player sprite — confirm licensing terms before treating this repo as fully clear for redistribution.
