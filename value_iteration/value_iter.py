from value_iteration.food_truck_env import FoodTruck
import numpy as np

np.set_printoptions(suppress=True)
epochs = 100
k = 0.1
gamma = 0.9
flg = 0


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
                if type(actions) == int:
                    next_rwd = env.get_delayed_rwd(current_state)
                    if hyperbolic:
                        val_est[current_state[0], current_state[1]] = hyperbolic_fcn(1) * next_rwd
                    else:
                        val_est[current_state[0], current_state[1]] = next_rwd
                    continue
                elif actions is None or env.terminal_state(current_state):
                    # If the action happens to be a wall block or it's inside a restaurant (which has 0 possible future
                    # reward) then don't do anything
                    continue
                possible_future_states = [current_state + env.get_change(act) for act in actions]
                if hyperbolic:
                    next_rwd_state = [(env.get_reward(next_state) + hyperbolic_fcn(1) * val_est[
                        next_state[0], next_state[1]], next_state) for next_state in
                                      possible_future_states]
                else:
                    next_rwd_state = [(env.get_reward(next_state) + gamma * val_est[
                        next_state[0], next_state[1]], next_state) for next_state in
                                      possible_future_states]
                max_rwd = next_rwd_state[0][0]
                max_next_state = next_rwd_state[0][1]
                for nrs in next_rwd_state:
                    if nrs[0] > max_rwd:
                        max_rwd = nrs[0]
                        max_next_state = nrs[1]
                if hyperbolic:
                    util_est = max_rwd
                else:
                    if current_state[0] == 5 and current_state[1] == 4:
                        print(current_state)
                        print("max_next_state", max_next_state)
                        print("possible_future_states", possible_future_states)
                        print(next_rwd_state)
                    util_est = max_rwd
                val_est[current_state[0], current_state[1]] = util_est

        print("\n", np.array2string(val_est[1:-1, 1:-1], precision=2))
        # print(val_est[1:-1, 1:-1].sum())
        # print(env.get_reward([5,4]))
        # print(val_est[4,4])


val_iters()
