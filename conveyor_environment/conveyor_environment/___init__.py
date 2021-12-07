from gym.envs.registration import register

register(
    id="conveyor_network_v0",
    entry_point="conveyor_environment.envs:ConveyorEnv_v0")
