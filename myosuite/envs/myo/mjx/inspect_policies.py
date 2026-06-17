import pickle
import jax
import jax.numpy as jp
import numpy as np

POLICY_A = "BaodingHold_64N_run1_BEST.pickle"
POLICY_B = "BaodingHold_64N_run1.pickle"

OUTPUT_FILE = "policy_nn_comparison.txt"


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_policy_nn(params):
    """
    Brax PPO saved params are:
        params[0] = observation normalizer
        params[1] = policy / actor neural network
        params[2] = value / critic neural network

    This script compares only params[1], the actor/policy NN.
    """
    policy_params = params[1]
    leaves, _ = jax.tree_util.tree_flatten(policy_params)
    arrays = [jp.asarray(x) for x in leaves if hasattr(x, "shape")]
    return arrays


def classify_array(index, array):

    shape = array.shape

    if len(shape) == 1:
        kind = "BIAS VECTOR"
        if index == 0:
            layer = "hidden layer 1 bias"
        elif index == 2:
            layer = "hidden layer 2 bias"
        elif index == 4:
            layer = "hidden layer 3 bias"
        elif index == 6:
            layer = "output layer bias"
        else:
            layer = "bias vector, exact layer uncertain"

        meaning = f"one bias value added to each of the {shape[0]} neurons/outputs"

    elif len(shape) == 2:
        kind = "WEIGHT MATRIX"
        if index == 1:
            layer = "input to hidden layer 1 weights"
        elif index == 3:
            layer = "hidden layer 1 to hidden layer 2 weights"
        elif index == 5:
            layer = "hidden layer 2 to hidden layer 3 weights"
        elif index == 7:
            layer = "hidden layer 3 to output layer weights"
        else:
            layer = "weight matrix, exact layer uncertain"

        meaning = (
            f"connects {shape[0]} input/previous-layer values "
            f"to {shape[1]} output neurons/values"
        )

    elif len(shape) == 0:
        kind = "SCALAR PARAMETER"
        layer = "scalar parameter"
        meaning = "single scalar value"

    else:
        kind = "OTHER PARAMETER"
        layer = "non-standard parameter"
        meaning = f"parameter with shape {shape}"

    return kind, layer, meaning


def main():
    params_a = load_policy(POLICY_A)
    params_b = load_policy(POLICY_B)

    arrays_a = flatten_policy_nn(params_a)
    arrays_b = flatten_policy_nn(params_b)

    if len(arrays_a) != len(arrays_b):
        raise ValueError(
            f"Different number of actor NN arrays: "
            f"{POLICY_A} has {len(arrays_a)}, "
            f"{POLICY_B} has {len(arrays_b)}"
        )

    total_values = 0
    total_abs_diff = 0.0
    total_sq_diff = 0.0
    max_abs_diff_global = 0.0

    per_array_summaries = []

    with open(OUTPUT_FILE, "w") as f:
        f.write("ACTOR / POLICY NEURAL NETWORK COMPARISON\n")
        f.write("=" * 80 + "\n")
        f.write(f"Policy A: {POLICY_A}\n")
        f.write(f"Policy B: {POLICY_B}\n")
        f.write("\n")
        f.write("Compared part: params[1] = actor / policy neural network\n")
        f.write("Ignored parts:\n")
        f.write("  params[0] = observation normalizer\n")
        f.write("  params[2] = value / critic neural network\n")
        f.write("\n")
        f.write(f"Number of actor NN arrays: {len(arrays_a)}\n")
        f.write("\n")

        for i, (a, b) in enumerate(zip(arrays_a, arrays_b)):
            a_np = np.array(a)
            b_np = np.array(b)

            if a_np.shape != b_np.shape:
                raise ValueError(
                    f"Shape mismatch in array {i}: "
                    f"{a_np.shape} vs {b_np.shape}"
                )

            diff = b_np - a_np

            array_l2_diff = float(np.sqrt(np.sum(diff ** 2)))
            array_mean_abs_diff = float(np.mean(np.abs(diff)))
            array_max_abs_diff = float(np.max(np.abs(diff)))

            total_values += diff.size
            total_abs_diff += np.sum(np.abs(diff))
            total_sq_diff += np.sum(diff ** 2)
            max_abs_diff_global = max(max_abs_diff_global, array_max_abs_diff)

            kind, layer, meaning = classify_array(i, a_np)

            per_array_summaries.append(
                {
                    "index": i,
                    "kind": kind,
                    "layer": layer,
                    "shape": a_np.shape,
                    "l2": array_l2_diff,
                    "mean_abs": array_mean_abs_diff,
                    "max_abs": array_max_abs_diff,
                }
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ACTOR NN ARRAY {i}\n")
            f.write("=" * 80 + "\n")
            f.write(f"type: {kind}\n")
            f.write(f"layer: {layer}\n")
            f.write(f"shape: {a_np.shape}\n")
            f.write(f"meaning: {meaning}\n")
            f.write("\n")
            f.write("DIFFERENCE SUMMARY FOR THIS ARRAY\n")
            f.write("-" * 80 + "\n")
            f.write(f"L2 difference: {array_l2_diff:.6e}\n")
            f.write(f"Mean absolute difference: {array_mean_abs_diff:.6e}\n")
            f.write(f"Max absolute difference: {array_max_abs_diff:.6e}\n")
            f.write("\n")

            f.write("VALUES IN POLICY A\n")
            f.write("-" * 80 + "\n")
            f.write(np.array2string(a_np, threshold=np.inf))
            f.write("\n\n")

            f.write("VALUES IN POLICY B\n")
            f.write("-" * 80 + "\n")
            f.write(np.array2string(b_np, threshold=np.inf))
            f.write("\n\n")

            f.write("DIFFERENCE B - A\n")
            f.write("-" * 80 + "\n")
            f.write(np.array2string(diff, threshold=np.inf))
            f.write("\n")

    global_l2_diff = float(np.sqrt(total_sq_diff))
    mean_abs_diff = float(total_abs_diff / total_values)

    print("Done.")
    print(f"Saved full actor NN comparison to: {OUTPUT_FILE}")
    print()
    print("GLOBAL ACTOR NN DIFFERENCE")
    print(f"Total compared values: {total_values}")
    print(f"Global L2 difference: {global_l2_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Max absolute difference: {max_abs_diff_global:.6e}")
    print()
    print("  Remember the interpretation:")
    print("  L2 difference = overall size of the change in this array.\n")
    print("  Mean absolute difference = average absolute change per value.\n")
    print("  Max absolute difference = largest single changed value in this array.\n")
    print()

if __name__ == "__main__":
    main()