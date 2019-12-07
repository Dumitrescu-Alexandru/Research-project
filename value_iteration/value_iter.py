from food_truck_env import FoodTruck
import numpy as np

np.set_printoptions(suppress=True)
no_value_fcns = 1000
epochs = 100
k = 0.5
gamma = 1
flg = 0


# approximate as many value functions for each coefficient

def hyperbolic_coefs():
    coeffs = []
    gamma_intervals = np.linspace(0, 1, no_value_fcns)
    gamma_intervals = gamma_intervals
    for i in range(1, no_value_fcns - 1):
        coeffs.append(
            ((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** ((1 / k) - 1)))
    coeffs = np.array(coeffs)
    return coeffs / sum(coeffs)


def create_multiple_fcns(state_space, vals=0):
    if type(vals) == int:
        return np.concatenate([np.zeros((state_space[0], state_space[1])) for _ in range(no_value_fcns - 2)]).reshape(
            state_space[0], state_space[1], -1)

    interm_vals = get_final_val_est(vals).reshape(state_space[0], state_space[1], 1)
    interm_vals = np.repeat(interm_vals, no_value_fcns - 2, axis=2)
    return interm_vals


def hyperbolic_est(val_est, rwd, next_state):
    return sum(hyperbolic_coefs() * val_est[next_state[0], next_state[1], :]) + rwd


def update_hyperbolic(val_est, next_state, max_next_rwd, current_state):
    gammas = np.linspace(0, 1, no_value_fcns)[1:no_value_fcns-1]
    val_est[current_state[0], current_state[1], :] = gammas * val_est[next_state[0], next_state[1],
                                                                          :] + max_next_rwd
    return val_est


def get_final_val_est(val_est):
    state_dims = val_est.shape[:-1]
    return np.sum(val_est.reshape(-1, no_value_fcns - 2) * hyperbolic_coefs(), axis=1).reshape(state_dims)


def val_iters(hyperbolic=True):
    env = FoodTruck()
    rows, cols = env.get_state_space()
    if hyperbolic:
        val_est = create_multiple_fcns(env.get_state_space())
    else:
        val_est = np.zeros(env.get_state_space())
    for ep in range(epochs):
        for i in range(1, rows):
            for j in range(1, cols):
                val_est = create_multiple_fcns(env.get_state_space(), vals=val_est)
                current_state = np.array([i, j])
                actions = env.possible_actions(current_state)
                if type(actions) == int:
                    next_rwd = env.get_delayed_rwd(current_state)
                    if hyperbolic:
                        val_est[current_state[0], current_state[1], :] = next_rwd
                    else:
                        val_est[current_state[0], current_state[1]] = next_rwd
                    continue
                elif actions is None or env.terminal_state(current_state):
                    # If the action happens to be a wall block or it's inside a restaurant (which has 0 possible future
                    # reward) then don't do anything
                    continue
                possible_future_states = [current_state + env.get_change(act) for act in actions]
                if hyperbolic:
                    next_rwd_state = [
                        (hyperbolic_est(val_est, env.get_reward(next_state), next_state), next_state)
                        for next_state in possible_future_states]
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
                    val_est = update_hyperbolic(val_est, max_next_state,
                                                env.get_reward((max_next_state[0], max_next_state[1])), current_state)
                else:
                    # if current_state[0] == 5 and current_state[1] == 4:
                    #     print(current_state)
                    #     print("max_next_state", max_next_state)
                    #     print("possible_future_states", possible_future_states)
                    #     print(next_rwd_state)
                    util_est = max_rwd
                    val_est[current_state[0], current_state[1]] = util_est
        if hyperbolic:
            # pass
            print("\n", np.array2string(get_final_val_est(val_est[1:-1, 1:-1]), precision=2))
            # print("\n", np.array2string(get_final_val_est(val_est), precision=2))
        else:
            print("\n", np.array2string(val_est[1:-1, 1:-1], precision=2))
        # print("\n", np.array2string(val_est, precision=2))
        # print(val_est[1:-1, 1:-1].sum())
        # print(env.get_reward([5,4]))
        # print(val_est[4,4])


val_iters()
