#!/usr/bin/python
import argparse
import csv
import glob
from pathlib import Path
from kr_cbs import KRCBSSolver
from kr_cbs_range import KRCBSRangeSolver
from ta_random import TaRandomSolver
from ta_distance import TaDistanceSolver
from ta_cbs import TACBSSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost

SOLVER = "KRCBS"


def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals


def _to_smart_xy(loc_rc):
    """Convert (row, col) to SMART (x, y)."""
    return loc_rc[1], loc_rc[0]


def _format_smart_path_point(loc_rc, t, path_format):
    row, col = loc_rc
    if path_format == "xy":
        return f"({col},{row},{t})"
    return f"({row},{col},{t})"


def _path_has_motion(path):
    if path is None or len(path) < 2:
        return False
    for t in range(1, len(path)):
        if path[t] != path[t - 1]:
            return True
    return False


def _path_relative_to_smart_dir(path_like):
    smart_dir = Path("smart").resolve()
    path_abs = Path(path_like).resolve()
    try:
        return path_abs.relative_to(smart_dir)
    except ValueError:
        return path_abs


def export_smart_files(instance_file, solver_name, k, my_map, starts, goals, paths, output_dir, path_format, num_agents=None):
    if paths is None:
        print(f"Skipping SMART export for {instance_file}: no paths found.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_indices = [i for i, path in enumerate(paths) if _path_has_motion(path)]
    if num_agents is not None:
        candidate_indices = candidate_indices[:num_agents]

    export_agents = len(candidate_indices)
    if export_agents <= 0:
        raise RuntimeError("No movable agents to export for SMART.")

    base = Path(instance_file).stem
    solver_tag = solver_name.lower().replace("-", "_")
    tag = f"{base}-{solver_tag}-k{k}"

    map_path = output_dir / f"{tag}.map"
    scen_path = output_dir / f"{tag}.scen"
    paths_path = output_dir / f"{tag}-paths_{path_format}.txt"

    # Write SMART map format.
    with open(map_path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {len(my_map)}\n")
        f.write(f"width {len(my_map[0])}\n")
        f.write("map\n")
        for r in range(len(my_map)):
            row = "".join("@" if my_map[r][c] else "." for c in range(len(my_map[0])))
            f.write(row + "\n")

    # Write SMART scen format (tab-separated columns).
    with open(scen_path, "w") as f:
        f.write("version 1\n")
        map_name = map_path.name
        width = len(my_map[0])
        height = len(my_map)
        for i, path_idx in enumerate(candidate_indices):
            sx, sy = _to_smart_xy(starts[path_idx])
            # For TA-based solvers, assigned targets can be a permutation.
            # Use path endpoint so .scen matches the actual executed plan.
            gx, gy = _to_smart_xy(paths[path_idx][-1])
            bucket = 0
            dist = len(paths[path_idx]) - 1
            f.write(f"{bucket}\t{map_name}\t{width}\t{height}\t{sx}\t{sy}\t{gx}\t{gy}\t{dist:.8f}\n")

    # Write SMART path format: Agent i:(x,y,t)->...
    with open(paths_path, "w") as f:
        for i, path_idx in enumerate(candidate_indices):
            path = paths[path_idx]
            points = "->".join(_format_smart_path_point(path[t], t, path_format) for t in range(len(path)))
            f.write(f"Agent {i}:{points}->\n")

    skipped_agents = len(paths) - len(candidate_indices)
    return map_path, scen_path, paths_path, export_agents, skipped_agents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--instance-csv', type=str, default=None,
                        help='CSV file with instance paths in the first column')
    parser.add_argument('--k', type=int, default=None,
                        help='The parameter for K-Robust KRCBS', required=True)
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--solver', type=str, required=True,
                        help='The solver to use (KRCBS, TA-RANDOM, TA-DISTANCE, TA-CBS)')
    parser.add_argument('--smart', action='store_true', default=False,
                        help='Export SMART simulator files (.map, .scen, path .txt)')
    parser.add_argument('--smart-output-dir', type=str, default='smart/generated',
                        help='Output directory for SMART files')
    parser.add_argument('--smart-path-format', choices=['xy', 'yx'], default='xy',
                        help='Path coordinate format for SMART path file (xy -> use --flip_coord=0, yx -> --flip_coord=1)')
    parser.add_argument('--smart-num-agents', type=int, default=None,
                        help='Limit number of agents exported for SMART files')

    args = parser.parse_args()

    if args.instance is None and args.instance_csv is None:
        raise RuntimeError("Provide --instance (glob pattern) or --instance-csv.")

    if args.instance is not None and args.instance_csv is not None:
        raise RuntimeError("Use only one of --instance or --instance-csv.")

    result_file = open("results.csv", "w", buffering=1)

    files = []
    if args.instance is not None:
        files = sorted(glob.glob(args.instance))
    else:
        with open(args.instance_csv, newline='') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row:
                    continue
                instance_path = row[0].strip()
                if instance_path:
                    files.append(instance_path)

    for file in files:

        print("***Import an instance***")
        my_map, starts, goals = import_mapf_instance(file)
        print_mapf_instance(my_map, starts, goals)

        if args.solver == "KRCBS":
            print("***Run KRCBS***")
            KRCBS = KRCBSSolver(my_map, starts, goals, k=args.k)
            paths = KRCBS.find_solution()
            
        elif args.solver == "KRCBS-RANGE":
            print("***Run KRCBS-RANGE***")
            KRCBS_range = KRCBSRangeSolver(my_map, starts, goals, k=args.k)
            paths = KRCBS_range.find_solution()

        elif args.solver == "TA-RANDOM":
            print("***Run TA-RANDOM***")
            ta_random = TaRandomSolver(my_map, starts, goals, k=args.k)
            paths = ta_random.find_solution()

        elif args.solver == "TA-DISTANCE":
            print("***Run TA-DISTANCE***")
            ta_distance = TaDistanceSolver(my_map, starts, goals, k=args.k)
            paths = ta_distance.find_solution()

        elif args.solver == "TA-CBS":
            print("***Run TA-CBS***")
            ta_cbs = TACBSSolver(my_map, starts, goals, k=args.k)
            paths = ta_cbs.find_solution()

        else:
            raise RuntimeError("Unknown solver!")

        cost = get_sum_of_cost(paths)
        result_file.write("{},{}\n".format(file, cost))

        if args.smart:
            export_info = export_smart_files(
                instance_file=file,
                solver_name=args.solver,
                k=args.k,
                my_map=my_map,
                starts=starts,
                goals=goals,
                paths=paths,
                output_dir=args.smart_output_dir,
                path_format=args.smart_path_format,
                num_agents=args.smart_num_agents,
            )
            if export_info is not None:
                map_path, scen_path, paths_path, export_agents, skipped_agents = export_info
                flip_coord = 0 if args.smart_path_format == "xy" else 1
                print("***SMART export***")
                print(f"map:  {map_path}")
                print(f"scen: {scen_path}")
                print(f"path: {paths_path}")
                if skipped_agents > 0:
                    print(f"note: skipped {skipped_agents} stationary agent(s) for SMART compatibility")
                map_in_smart = _path_relative_to_smart_dir(map_path)
                scen_in_smart = _path_relative_to_smart_dir(scen_path)
                path_in_smart = _path_relative_to_smart_dir(paths_path)
                print("Run from smart/ directory:")
                print(
                    "python3 run_sim.py "
                    f"--map_name={map_in_smart} "
                    f"--scen_name={scen_in_smart} "
                    f"--num_agents={export_agents} "
                    f"--path_filename={path_in_smart} "
                    f"--flip_coord={flip_coord} "
                    "--headless=True"
                )
                print("Full command:")
                print(
                    "cd smart && "
                    "python3 run_sim.py "
                    f"--map_name={map_in_smart} "
                    f"--scen_name={scen_in_smart} "
                    f"--num_agents={export_agents} "
                    f"--path_filename={path_in_smart} "
                    f"--flip_coord={flip_coord} "
                    "--headless=True"
                )

        if not args.batch:
            print("***Test paths on a simulation***")
            animation = Animation(my_map, starts, goals, paths, k=args.k)
            # animation.save("output.mp4", 1.0)
            animation.show()
    result_file.close()
