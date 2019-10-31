from food_truck_env import FoodTruck
import numpy as np

epochs = 100
k = 0.1
gamma = 0.9


def hyperbolic_fcn(d):
    return 1 / (1 + k * d)


def val_iters(hyperbolic=True):
    env = FoodTruck()
    rows, cols = env.get_state_space()
    val_est = np.zeros(env.get_state_space())
    for ep in range(epochs):
        for i in range(1, rows):
            for j in range(1, cols):
                current_state = np.array([i, j])
                actions = env.possible_actions(current_state)
                if actions is None or env.terminal_state(current_state):
                    # If the action happens to be a wall block or it's inside a restaurant (which has 0 possible future
                    # reward) then don't do anything
                    continue
                possible_future_states = [current_state + env.get_change(act) for act in actions]
                next_rwd_state = [(env.get_reward(next_state), next_state) for next_state in possible_future_states]
                max_rwd = next_rwd_state[0][0]
                max_next_state = next_rwd_state[0][1]
                for nrs in next_rwd_state:
                    if nrs[0] > max_rwd:
                        max_rwd = nrs[0]
                        max_next_state = nrs[1]
                if hyperbolic:
                    if env.terminal_state(max_next_state):
                        util_est = hyperbolic_fcn(1) * max_rwd + val_est[
                            max_next_state[0], max_next_state[1]] * hyperbolic_fcn(1)
                    else:
                        util_est = max_rwd
                else:
                    if env.terminal_state(max_next_state):
                        util_est = max_rwd
                    else:
                        util_est = max_rwd + val_est[max_next_state[0], max_next_state[1]] * gamma
                val_est[current_state[0], current_state[1]] = util_est
        print("\n", val_est[1:-1, 1:-1])


val_iters()
