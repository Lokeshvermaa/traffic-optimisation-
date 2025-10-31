def get_state(traci):
    """
    Returns a simplified state representation — e.g. total queue length.
    """
    total_waiting = 0
    for lane_id in traci.lane.getIDList():
        total_waiting += traci.lane.getLastStepHaltingNumber(lane_id)
    return str(total_waiting)  # simple string state


def get_reward(traci):
    """
    Reward is the negative of total waiting vehicles — lower waiting = better.
    """
    total_waiting = 0
    for lane_id in traci.lane.getIDList():
        total_waiting += traci.lane.getLastStepHaltingNumber(lane_id)
    return -total_waiting
