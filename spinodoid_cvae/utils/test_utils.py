# utils/test_utils.py

def run_max_inverse_design(
    C_true,
    S_true,
    max_repo_dir="max_repo",
    model_path="max_repo/results/models/dataset_train_x0075/tvs_PEFNN.h5",
    pass_threshold=0.08,
    show=True
):
    """
    Runs Max's inverse design on the given C_true and returns:
      - df (one-row table): Ŝ, ΔS, error, status, interesting
      - S_star (np.ndarray, shape (4,))
      - Phi_star (np.ndarray, shape (3,))
      - final_loss (float, normalized C-error)
    """
    import os, sys, numpy as np, pandas as pd, tensorflow as tf
    import importlib

    # make Max's modules importable
    src_path = os.path.join(max_repo_dir, "src")
    if max_repo_dir not in sys.path: sys.path.append(max_repo_dir)
    if src_path     not in sys.path: sys.path.append(src_path)

    inv_mod   = importlib.import_module("src.inverse_design")
    math_mod  = importlib.import_module("util.mathops")
    inverse_design             = inv_mod.inverse_design
    apply_rotation             = math_mod.apply_rotation
    euclidian_distance         = math_mod.euclidian_distance
    angles_to_rotation_matrix  = math_mod.angles_to_rotation_matrix

    # load TF model (compile=False suppresses compile warning)
    tf_model = tf.keras.models.load_model(model_path, compile=False)

    # objective (normalized C-tensor error, uses optimized rotation Q)
    C_ref = tf.convert_to_tensor(C_true, dtype=tf.float32)

    @tf.function
    def objective(S, Q):
        # S: (4,)   Q: (3,3)
        S_in   = tf.reshape(S, [1, 1, -1])             # (1,1,4)
        C_pred = tf_model(S_in, training=False)[0, 0]  # (3,3,3,3)
        C_rot  = apply_rotation(C_pred, Q)
        num = euclidian_distance(C_rot, C_ref, 4)**2
        den = euclidian_distance(C_ref, tf.zeros_like(C_ref), 4)
        return num / den

    # run inverse design (no constraints)
    S_star_tf, Phi_star_tf = inverse_design(objective, constraints=[])
    S_star   = S_star_tf.numpy()   # (theta1, theta2, theta3, rho)
    Phi_star = Phi_star_tf.numpy() # (phi, omega, eps)

    # evaluate final normalized error with the optimized rotation
    Q_star     = angles_to_rotation_matrix(tf.constant(Phi_star, dtype=tf.float32))
    final_loss = float(objective(tf.constant(S_star, tf.float32), Q_star).numpy())

    # tiny local formatter (keeps my table style)
    def _format_array(arr, precision=4):
        arr = np.asarray(arr, dtype=float).flatten()
        return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"

    # one-row table (matches your model tables)
    df = pd.DataFrame([{
        "Ŝ":          _format_array(S_star),
        "ΔS":          _format_array(S_star - S_true),
        "error":       f"{final_loss*100:.2f}%",
        "status":      "✅" if final_loss < pass_threshold else "❌ FAIL",
        "interesting": "✅" if (np.abs(S_star[:3] - S_true[:3]) >= 5.0).any() else "❌",
    }])

    if show:
        print("\n⚪ Max inverse design — 1 candidate")
        from IPython.display import display
        display(df)

    return df, S_star, Phi_star, final_loss
