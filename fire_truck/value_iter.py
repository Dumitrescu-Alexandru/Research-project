from matplotlib import colors
from food_truck_env import FoodTruck
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
no_value_fcns = 1000
epochs = 10
k = 0.9
gamma = np.sort(np.random.uniform(0, 1, no_value_fcns))
# gamma = np.linspace(0, 1, no_value_fcns)
flg = 0


# sample gammas for the intervals instead of linspacing
# try it asynch (update on the non-already updated val_functions)

def hyperbolic_coefs():
    coeffs = []
    for i in range(1, no_value_fcns - 1):
        coeffs.append(
            ((gamma[i + 1] - gamma[i]) * (1 / k) * gamma[i] ** ((1 / k) - 1)))
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


def update_hyperbolic(val_est_, next_state, max_next_rwd, current_state):
    # gammas = np.linspace(0, 1, no_value_fcns)[1:no_value_fcns - 1]
    gammas = gamma[1:-1]
    val_est_[current_state[0], current_state[1], :] = gammas * val_est_[next_state[0], next_state[1],
                                                               :] + max_next_rwd
    return val_est_


def get_final_val_est(val_est):
    state_dims = val_est.shape[:-1]
    return np.sum(val_est.reshape(-1, no_value_fcns - 2) * hyperbolic_coefs(), axis=1).reshape(state_dims)


def val_iters(hyperbolic=True, alternative_impl=True, rwd_on_exit=True):
    env = FoodTruck()
    rows, cols = env.get_state_space()
    # alternative restaurant states
    # restaurant_states = {(8, 1): -11.01, (3, 3): -11.01, (1, 5): 21.01, (6, 6): -0.01}
    restaurant_states = {(8, 1): -10.01, (3, 3): -10.01, (1, 5): 20.01, (6, 6): -0.01}
    if hyperbolic:
        val_est = create_multiple_fcns(env.get_state_space())
    else:
        val_est = np.zeros(env.get_state_space())
    convergence_copy = np.copy(val_est)
    for ep in range(epochs):
        update_copy = np.copy(val_est)
        # print("On ep", ep, np.sum(np.abs(convergence_copy-val_est)))
        # use this to check prev_step_val_est - prev_to_prev_step_val_est val difference
        convergence_copy = np.copy(val_est)
        for i in range(1, rows):
            for j in range(1, cols):
                current_state = np.array([i, j])
                actions = env.possible_actions(current_state)
                if type(actions) == int:
                    current_rwd = env.get_reward(current_state)
                    next_rwd = env.get_delayed_rwd(current_state)
                    if hyperbolic:
                        if rwd_on_exit:
                            current_rwd = np.repeat(current_rwd, no_value_fcns - 2)
                            next_state_val = np.repeat(restaurant_states[tuple(current_state)], no_value_fcns - 2)
                            # gammas = np.linspace(0, 1, no_value_fcns)[1:no_value_fcns - 1]
                            gammas = gamma[1:-1]
                            val_est[current_state[0], current_state[1], :] = current_rwd + next_state_val * gammas
                        else:
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
                    if alternative_impl:
                        # gammas = np.linspace(0, 1, no_value_fcns)[1:no_value_fcns - 1]
                        gammas = gamma[1:-1]
                        next_state_vals = np.concatenate(
                            [update_copy[n_s[0], n_s[1]] for n_s in possible_future_states])
                        discounted_ns = next_state_vals.reshape(-1, no_value_fcns - 2) * gammas
                        if rwd_on_exit:
                            rwds = np.array([env.get_reward(current_state) for _ in possible_future_states])
                            rwds = rwds.reshape(-1, 1)
                            rwds = np.repeat(rwds, no_value_fcns - 2, axis=1)
                        else:
                            rwds = np.array([env.get_reward(ns) for ns in possible_future_states])
                            rwds = rwds.reshape(-1, 1)
                            rwds = np.repeat(rwds, no_value_fcns - 2, axis=1)
                        max_ns_vals = np.max(discounted_ns + rwds, axis=0)
                        val_est[current_state[0], current_state[1], :] = max_ns_vals
                    else:
                        next_rwd_state = [
                            (hyperbolic_est(update_copy, env.get_reward(next_state), next_state), next_state)
                            for next_state in possible_future_states]
                else:
                    gammas = gamma[1:-1]
                    next_rwd_state = [(env.get_reward(next_state) + gammas * update_copy[
                        next_state[0], next_state[1]], next_state) for next_state in
                                      possible_future_states]
                if not alternative_impl:
                    max_rwd = next_rwd_state[0][0]
                    max_next_state = next_rwd_state[0][1]
                    for nrs in next_rwd_state:
                        if nrs[0] > max_rwd:
                            max_rwd = nrs[0]
                            max_next_state = nrs[1]
                if hyperbolic and not alternative_impl:
                    val_est = update_hyperbolic(update_copy, max_next_state,
                                                env.get_reward((max_next_state[0], max_next_state[1])), current_state)
                elif not hyperbolic:
                    # if current_state[0] == 5 and current_state[1] == 4:
                    #     print(current_state)
                    #     print("max_next_state", max_next_state)
                    #     print("possible_future_states", possible_future_states)
                    #     print(next_rwd_state)
                    util_est = max_rwd
                    val_est[current_state[0], current_state[1]] = util_est
        if hyperbolic:
            # pass
            final_val_est = get_final_val_est(val_est[1:-1, 1:-1])
            # print("\n", np.array2string(get_final_val_est(val_est[1:-1, 1:-1]), precision=2))
            print(final_val_est[6, 2], final_val_est[5, 3])
            print("It will go left" if final_val_est[6, 2] > final_val_est[5, 3] else "it will go up")
            # print(final_val_est[6, 2], final_val_est[5, 3])

            # print("\n", np.array2string(get_final_val_est(val_est), precision=2))
        else:
            print("\n", np.array2string(val_est[1:-1, 1:-1], precision=2))
    data = get_final_val_est(val_est[1:-1, 1:-1])
    plt.rc('font', size=8)  # controls default text sizes
    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=13)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)  # legend fontsize
    plt.rc('figure', titlesize=13)  # fontsize of the figure title
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    y_positions = np.array(list(range(7, -1, -1)))
    x_positions = np.array(list(range(0, 6)))
    for y_, y in enumerate(y_positions):
        for x_, x in enumerate(x_positions):
            color = "black"
            label = round(data[y, x], 3)
            if data[y, x] == 0:
                val_est[y+1,x+1] = -2
                label = "Wall "
                color = "white"
            elif y == 7 and x == 0 or x == 2 and y == 2:
                label = "Dnt "  + str(round(data[y, x], 2))
            elif y == 5 and x == 5:
                label = "Ndl "  + str(round(data[y, x],2))
            elif y == 0 and x == 4:
                label = "Veg " + str(round(label, 2))
            else:
                label = str(label)
                color = "black"
            ax.text(x, y, label, color=color, ha='center', va='center')
    plt.imshow(get_final_val_est(val_est[1:-1, 1:-1]), cmap="gray")
    # print(env.ft_map)
    # ax.scatter(7, 0, marker="s",  c="black", label="wall")
    # ax.scatter(7, 0, marker="s",  c="firebrick", label="veggie restaurant")
    # ax.scatter(7, 0, marker="s",  c="rosybrown", label="noodle restaurant")
    # ax.scatter(7, 0, marker="s",  c="sienna", label="donut restaurant")
    # ax.scatter(7, 0, marker="s",  c="lightgrey", label="accessible path")
    # image_to_show = env.ft_map[1:-1, 1:-1]
    # image_to_show[7, 0] = 2
    # image_to_show[0, 4] = 6
    # image_to_show[5, 5] = 4
    # print(image_to_show)
    # cmap = colors.ListedColormap(['black', 'lightgrey', 'sienna', 'rosybrown', 'firebrick'])
    # ax.imshow(env.ft_map[1:-1, 1:-1], label="Value of ", cmap=cmap)
    # ax.legend(loc='upper left', framealpha=0.4)
    # ax.legend(bbox_to_anchor=(1.05, 0.65), loc='upper left')
    plt.title("Hyperbolic agent in Food Truck")

    plt.show()


val_iters()
