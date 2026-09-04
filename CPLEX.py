from docplex.mp.model import Model
import importlib
import config


# Set this to 9000 for large-scale instances.
CPLEX_TIME_LIMIT = 3600


def reload_config():
    try:
        importlib.reload(config)
        return (config.N, config.T, config.B,
                config.r_j, config.seta, config.c,
                config.t_k, config.p_j,
                config.L_k, config.S_k, config.CP_k)
    except (ImportError, AttributeError):
        print("Failed to load config.py.")
        raise


def main():
    m = Model(name="mip_with_batches")

    m.parameters.timelimit = CPLEX_TIME_LIMIT

    N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k = reload_config()

    X = m.binary_var_dict(keys=[(b, k, j) for b in range(B) for k in range(T) for j in range(N)], name="X")
    Y = m.binary_var_dict(keys=[(b, k)    for b in range(B) for k in range(T)], name="Y")
    Z = m.binary_var_dict(keys=[k        for k in range(T)], name="Z")

    C_bk = m.continuous_var_dict(keys=[(b, k) for b in range(B) for k in range(T)], name="C_bk", lb=0)
    P_bk = m.continuous_var_dict(keys=[(b, k) for b in range(B) for k in range(T)], name="P_bk", lb=0)
    S_bk = m.continuous_var_dict(keys=[(b, k) for b in range(B) for k in range(T)], name="S_bk", lb=0)
    P_k  = m.continuous_var_dict(keys=[k for k in range(T)], name="P_k", lb=0)

    m.minimize(
        m.sum(P_k[k] * t_k[k] for k in range(T)) +
        m.sum(CP_k[k] * Z[k] for k in range(T))
    )

    for j in range(N):
        m.add_constraint(
            m.sum(X[b, k, j] for b in range(B) for k in range(T)) == 1,
            ctname=f"assign_{j}"
        )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                m.sum(X[b, k, j] for j in range(N)) <= c,
                ctname=f"batch_cap_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                P_bk[b, k] >= m.sum(X[b, k, j] * p_j[j] for j in range(N)),
                ctname=f"P_lower_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                Y[b, k] >= m.sum(X[b, k, j] for j in range(N)) / N,
                ctname=f"Y_active_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            for j in range(N):
                m.add_constraint(
                    S_bk[b, k] >= X[b, k, j] * r_j[j],
                    ctname=f"release_{b}_{k}_{j}"
                )

    for b in range(1, B):
        for k in range(T):
            m.add_constraint(
                S_bk[b, k] >= C_bk[b-1, k],
                ctname=f"seq_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                C_bk[b, k] >= S_bk[b, k] + seta * Y[b, k] + P_bk[b, k],
                ctname=f"C_def_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                S_bk[b, k] >= S_k[k],
                ctname=f"min_start_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                C_bk[b, k] - S_k[k] <= L_k[k],
                ctname=f"deadline_{b}_{k}"
            )

    for b in range(B):
        for k in range(T):
            m.add_constraint(
                P_k[k] >= C_bk[b, k] - S_k[k],
                ctname=f"P_k_cover_{b}_{k}"
            )

    for k in range(T):
        m.add_constraint(
            Z[k] >= m.sum(Y[b, k] for b in range(B)) / N,
            ctname=f"Z_usage_{k}"
        )

    print("Solving the MILP model...")
    msol = m.solve(log_output=True)

    if msol:
        print(f"Feasible solution found. Objective value = {m.objective_value:.2f}")
    else:
        print("Solver status:", m.solve_details.status)


if __name__ == "__main__":
    main()
