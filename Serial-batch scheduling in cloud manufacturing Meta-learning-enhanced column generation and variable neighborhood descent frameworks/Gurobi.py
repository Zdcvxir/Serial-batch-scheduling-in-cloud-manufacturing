from gurobipy import *

def reload_config():
    try:
        import importlib
        import config
        importlib.reload(config)
        return (config.N, config.T, config.B,
                config.r_j, config.seta, config.c,
                config.t_k, config.p_j,
                config.L_k, config.S_k, config.CP_k)
    except (ImportError, AttributeError):
        print('config.py not found')
    
def main():
    m = Model("mip")
    m.setParam('TimeLimit', 60 * 60)

    N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k = reload_config()

    X = m.addVars(B, T, N, vtype=GRB.BINARY, name="X")
    Y = m.addVars(B, T, vtype=GRB.BINARY, name="Y")
    Z = m.addVars(T, vtype=GRB.BINARY, name="Z")
    C_bk = m.addVars(B, T, vtype=GRB.CONTINUOUS, name="C_bk")
    P_bk = m.addVars(B, T, vtype=GRB.CONTINUOUS, name="P_bk")
    S_bk = m.addVars(B, T, vtype=GRB.CONTINUOUS, name="S_bk")
    P_k = m.addVars(T, vtype=GRB.CONTINUOUS, name="P_k")

    # 1
    m.setObjective(sum(P_k[k] * t_k[k] for k in range(T)) + sum(CP_k[k] * Z[k] for k in range(T)), GRB.MINIMIZE)
  
    # 2
    for j in range(N):
        m.addConstr(sum(X[b, k, j] for k in range(T) for b in range(B)) == 1)

    # 3
    for b in range(B):
        for k in range(T):
            m.addConstr(sum(X[b, k, j] for j in range(N)) <= c)

    # 4
    for b in range(B):
        for k in range(T):
            m.addConstr(P_bk[b, k] >= sum(X[b, k, j] * p_j[j] for j in range(N)))

    # 5
    for b in range(B):
        for k in range(T):
            m.addConstr(Y[b, k] >= P_bk[b, k] / sum(p_j[j] for j in range(N)))

    # 6
    for b in range(B):
        for k in range(T):
            for j in range(N):
                m.addConstr(S_bk[b, k] >= X[b, k, j] * r_j[j])

    # 7
    for b in range(1, B):
        for k in range(T):
            m.addConstr(S_bk[b, k] >= C_bk[b-1, k])

    # 8
    for b in range(B):
        for k in range(T):
            m.addConstr(C_bk[b, k] >= S_bk[b, k] + seta * Y[b, k] + P_bk[b, k]) 
    
    # 9
    for b in range(B):
        for k in range(T):
            m.addConstr(S_bk[b, k] >= S_k[k])   

    # 10
    for b in range(B):
        for k in range(T):
            m.addConstr(C_bk[b, k] - S_k[k] <= L_k[k])

    # 11
    for b in range(B):
        for k in range(T):
            m.addConstr(P_k[k] >= C_bk[b, k] - S_k[k] )

    # 12
    for k in range(T):
        m.addConstr(Z[k] >= sum(Y[b, k] for b in range(B)) / N) 

    m.optimize()

    # if m.status == GRB.OPTIMAL:
    #     print('Optimal Solution:', m.objVal)
    #     for var in m.getVars():
    #         print(f"{var.varName} = {var.x}")

main()
