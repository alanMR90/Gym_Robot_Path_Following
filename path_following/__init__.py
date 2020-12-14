import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Laser2DLine-v0',
    entry_point='path_following.envs:Laser2DLine'
)

register(
    id='Laser2DPoint-v0',
    entry_point='path_following.envs:Laser2DPoint'
)

register(
    id='Tool5D-v0',
    entry_point='path_following.envs:Tool5D'
)
