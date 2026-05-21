from anastruct import SystemElements

def run_simulation(load_value, state=None, visualize=False):
    ss = SystemElements()

    # Beam
    ss.add_element(location=[[0, 0], [5, 0]])
    ss.add_element(location=[[5, 0], [10, 0]])

    # Supports
    ss.add_support_hinged(node_id=1)
    ss.add_support_hinged(node_id=3)

    # Load
    ss.point_load(node_id=2, Fy=-abs(load_value) * 50)

    # Solve
    ss.solve()

    # Visualization (FINAL FIX)
    if visualize:
        print(f"\n===TWIN ===")
        print(f"State: {state} | Load: {load_value:.2f}")

        ss.show_displacement()

    # Output
    displacement = ss.get_node_displacements(node_id=2)
    return float(displacement['uy'])