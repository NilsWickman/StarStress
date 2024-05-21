from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from observations import (get_observation, to_tensor, to_cuda,
                          load_global_stats, extend_batch,
                          concat_along_axis_tensor, concat_lstm_hidden,
                          interface_format)
import os
import sys
from absl import flags

bot_type = Bot(Race.zerg, Difficulty.very_easy)
VISUALIZE = False
FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])


env = SC2Env(
                map_name="Acropolis",
                players=[Agent(Race.protoss, "StarTrain"), bot_type],
                agent_interface_format=interface_format,
                visualize=VISUALIZE,
                step_mul=1,
                ensure_available_actions=False,
                realtime=False,
                game_steps_per_episode=0,
                save_replay_episodes=1,
                score_index=-1,
                replay_dir=os.path.abspath("showcase_replays")
            )

env.step(0)