import sys
import re
from collections import deque


class verilogFuncs:
    # ------------------------------------------------------------------ #
    #  Primitive‑level simulation tables
    # ------------------------------------------------------------------ #
    primitive_simulators = {
        "AND": lambda inp: {"Y": inp.get("A", 0) & inp.get("B", 0)},
        "INV": lambda inp: {"Y": 0 if inp.get("A", 0) else 1},
        "XOR": lambda inp: {"Y": inp.get("A", 0) ^ inp.get("B", 0)},
        "FA":  lambda inp: verilogFuncs.simulate_full_adder(inp),
        "HA":  lambda inp: verilogFuncs.simulate_half_adder(inp),
    }

    primitive_ports = {
        "AND": {"inputs": ["A", "B"],          "outputs": ["Y"]},
        "INV": {"inputs": ["A"],               "outputs": ["Y"]},
        "XOR": {"inputs": ["A", "B"],          "outputs": ["Y"]},
        "FA":  {"inputs": ["A", "B", "CI"],    "outputs": ["SN", "CON"]},
        "HA":  {"inputs": ["A", "B"],          "outputs": ["SN", "CON"]},
    }

    # ------------------------------------------------------------------ #
    #  Primitive helpers (FA / HA logic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def simulate_full_adder(inp):
        A, B, CI = inp.get("A", 0), inp.get("B", 0), inp.get("CI", 0)
        SN  = (A ^ B) ^ CI
        CON = 1 if (A + B + CI) >= 2 else 0
        return {"SN": SN, "CON": CON}

    @staticmethod
    def simulate_half_adder(inp):
        A, B = inp.get("A", 0), inp.get("B", 0)
        return {"SN": A ^ B, "CON": A & B}

    # ------------------------------------------------------------------ #
    #  Verilog parser (hierarchical, minimal)
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_hierarchical_verilog(path):
        """
        Parses a Verilog file into a module dictionary.
        Correctly extracts ALL inputs/outputs from the module header
        *and* from body‑level declarations such as
            input  a, b;
            output y;
        """
        modules = {}
        current = None
        lines = [ln.rstrip() for ln in open(path)]

        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if not ln or ln.startswith("//"):
                i += 1
                continue

            # ---- module header ------------------------------------------------
            if ln.startswith("module"):
                # gather full header line (it might span multiple lines)
                hdr = ln
                while ";" not in hdr:
                    i += 1
                    hdr += " " + lines[i].strip()

                m = re.match(r"module\s+(\S+)\s*\(([^)]*)\)\s*;", hdr)
                if m:
                    name      = m.group(1)
                    port_list = [p.strip() for p in m.group(2).split(",")]

                    ports, inputs, outputs = [], [], []
                    direction = None

                    # iterate through comma‑separated items,
                    # switching direction on seeing 'input'/'output'
                    for item in port_list:
                        parts = item.split()
                        if parts[0] == "input" and len(parts) == 2:
                            direction = "input"
                            sig = parts[1]
                        elif parts[0] == "output" and len(parts) == 2:
                            direction = "output"
                            sig = parts[1]
                        else:
                            sig = parts[-1]  # inherit previous direction

                        ports.append(sig)
                        if direction == "input":
                            inputs.append(sig)
                        elif direction == "output":
                            outputs.append(sig)

                    modules[name] = {
                        "name": name,
                        "ports": ports,
                        "inputs": inputs,
                        "outputs": outputs,
                        "wires": [],
                        "regs": [],
                        "assigns": [],
                        "instances": []
                    }
                    current = modules[name]
                i += 1
                continue

            # ---- endmodule ----------------------------------------------------
            if ln.startswith("endmodule"):
                current = None
                i += 1
                continue

            # ---- inside module body -------------------------------------------
            if current:
                def pull(keyword):
                    blk = ln
                    while ";" not in blk:
                        i += 1
                        blk += " " + lines[i].strip()
                    return [s.strip() for s in
                            blk.replace(keyword, "").replace(";", "").split(",")
                            if s.strip()]

                if ln.startswith("wire"):
                    current["wires"].extend(pull("wire"))
                elif ln.startswith("reg"):
                    current["regs"].extend(pull("reg"))

                # 🔹 New: body‑level input/output declarations
                elif ln.startswith("input"):
                    current["inputs"].extend(pull("input"))
                elif ln.startswith("output"):
                    current["outputs"].extend(pull("output"))
                # 🔹 End new lines

                elif ln.startswith("assign"):
                    blk = ln
                    while ";" not in blk:
                        i += 1
                        blk += " " + lines[i].strip()
                    current["assigns"].extend([
                        x.strip() for x in
                        blk.replace("assign", "").replace(";", "").split(",")
                        if x.strip()
                    ])
                else:
                    # instance instantiation
                    inst = ln
                    while ";" not in inst and i + 1 < len(lines):
                        i += 1
                        inst += " " + lines[i].strip()
                    m2 = re.match(r"(\S+)\s+(\S+)\s*\((.*)\)\s*;", inst)
                    if m2:
                        mtype, iname, body = m2.group(1), m2.group(2), m2.group(3)
                        if "." in body:  # named connections
                            pairs = re.findall(r"\.(\w+)\s*\(\s*([^)]+)\s*\)", body)
                            conns = [{"port": p, "signal": s.strip()} for p, s in pairs]
                        else:            # positional
                            conns = [{"signal": s.strip()} for s in body.split(",")]
                        current["instances"].append({
                            "module": mtype,
                            "instance_name": iname,
                            "connections": conns
                        })
                i += 1
                continue

            # outside all modules
            i += 1

        return modules

    # ------------------------------------------------------------------ #
    #  Top‑module finder & input helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def find_top_module(mods):
        child = {inst["module"] for m in mods.values() for inst in m["instances"]}
        return next((m for m in mods if m not in child), None)

    @staticmethod
    def get_input_values(inp_list):
        if not inp_list:
            return {}
        vals = {}
        choose = input("Load inputs from .txt? (y/n): ").strip().lower()
        if choose.startswith("y"):
            fn = input("file name: ").strip()
            try:
                bits = re.sub(r"[,\\s]", "", open(fn).read())
                if len(bits) != len(inp_list):
                    raise ValueError("bit‑count mismatch")
                vals = {sig: int(bits[i]) for i, sig in enumerate(inp_list)}
                print("Loaded from file.")
                return vals
            except Exception as e:
                print("Error:", e, "— switching to manual entry")
        print("Enter each input (0/1):")
        for sig in inp_list:
            v = ""
            while v not in ("0", "1"):
                v = input(f"{sig}: ").strip()
            vals[sig] = int(v)
        return vals

    # ------------------------------------------------------------------ #
    #  Recursive hierarchical simulation (fixed debug flag)
    # ------------------------------------------------------------------ #
    @staticmethod
    def simulate_module(mods, name, in_vals, sigs, scope="", debug=False):
        """
        Recursively simulate module name.
        - mods     : dict of parsed modules
        - in_vals  : {port_name: bit_value}
        - sigs     : global dict to fill with scoped signals
        - scope    : hierarchical prefix
        - debug    : if True, prints intermediate steps
        """
        # leaf primitive?
        if name in verilogFuncs.primitive_simulators:
            pref = scope + name + "_"
            for out, val in verilogFuncs.primitive_simulators[name](in_vals).items():
                sigs[pref + out] = val
                if debug:
                    print(f"[PRIM] {pref + out} = {val}")
            return

        mod   = mods[name]
        pref  = scope + name + "_"
        local = {p: in_vals.get(p, 0) for p in mod["inputs"]}

        # recurse into each instance
        for inst in mod["instances"]:
            mtype, iname = inst["module"], inst["instance_name"]
            inst_scope   = pref + iname + "_"
            ports        = mods.get(mtype, {}).get("ports", [])
            child_in     = {}

            # collect inputs for this child
            for idx, conn in enumerate(inst["connections"]):
                role = conn.get("port", ports[idx] if idx < len(ports) else None)
                if role in mods.get(mtype, {}).get(
                        "inputs",
                        verilogFuncs.primitive_ports.get(mtype, {}).get("inputs", [])):
                    child_in[role] = local.get(conn["signal"], 0)

            # simulate the child
            verilogFuncs.simulate_module(mods, mtype, child_in, sigs,
                                         inst_scope, debug)

            # map child outputs back into our local signals
            outs = mods.get(mtype, {}).get(
                "outputs",
                verilogFuncs.primitive_ports.get(mtype, {}).get("outputs", []))
            for idx, conn in enumerate(inst["connections"]):
                role = conn.get("port", ports[idx] if idx < len(ports) else None)
                if role in outs:
                    local[conn["signal"]] = sigs.get(
                        inst_scope + mtype + "_" + role, 0)

        # evaluate assign statements
        for stmt in mod["assigns"]:
            if "=" in stmt:
                lhs, rhs = (x.strip() for x in stmt.split("=", 1))
                local[lhs] = verilogFuncs.evaluate_expression(rhs, local)

        # publish locals into global sigs dict
        for sig, val in local.items():
            sigs[pref + sig] = val
            if debug and sig in mod["outputs"]:
                print(f"[OUT ] {pref + sig} = {val}")

    # ------------------------------------------------------------------ #
    #  Simple expression evaluator
    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_expression(expr, nets):
        """
        Replace every token (signal name) in expr with its numeric value,
        then eval the boolean/arithmetic expression and return 0/1.
        """
        def repl(m):
            return str(nets.get(m.group(0), 0))

        # \b ensures we match whole words only
        expr_filled = re.sub(r"\b\w+\b", repl, expr)
        try:
            return int(bool(eval(expr_filled)))
        except Exception as e:
            print(f"[DEBUG] Error evaluating '{expr}': {e}")
            return 0

    # --------------- more helper routines follow in the second half ---------------
    # ------------------------------------------------------------------ #
    #  Label helper for boxed views
    # ------------------------------------------------------------------ #
    @staticmethod
    def box_label(inst_name, mod_type):
        nice = {"FA": "Full Adder", "HA": "Half Adder"}.get(mod_type, mod_type)
        return f"{inst_name} ({nice})"

    # ------------------------------------------------------------------ #
    #  Generic primitive boxed view (AND / INV / XOR / etc.)
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_primitive_dot(dot_file, gate_type, inst_name, port_info):
        ins  = verilogFuncs.primitive_ports[gate_type]["inputs"]
        outs = verilogFuncs.primitive_ports[gate_type]["outputs"]

        with open(dot_file, "w") as f:
            f.write("digraph PRIM {\n  rankdir=LR;\n")
            f.write(f'  subgraph cluster_p {{ label="{verilogFuncs.box_label(inst_name, gate_type)}"; '
                    "style=rounded;\n")
            f.write(f'    "CELL" [label="{gate_type}", shape=box, style=filled, fillcolor=white];\n')
            f.write("  }\n")
            # inputs
            for role in ins:
                sig, val = port_info[role]
                f.write(f'  "{sig}" [label="{sig}\\n({role}={val})", '
                        'shape=circle, style=filled, fillcolor=lightblue];\n')
                f.write(f'  "{sig}" -> "CELL";\n')
            # outputs
            for role in outs:
                sig, val = port_info[role]
                f.write(f'  "{sig}_OUT" [label="{sig}\\n({role}={val})", '
                        'shape=circle, style=filled, fillcolor=red];\n')
                f.write(f'  "CELL" -> "{sig}_OUT";\n')
            f.write("}\n")

    # ------------------------------------------------------------------ #
    #  Full‑Adder boxed view
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_FA_dot(dot_file, inst_name, port_info):
        (sigA, A), (sigB, B), (sigCI, CI) = (
            port_info["A"], port_info["B"], port_info["CI"]
        )
        XOR1 = A ^ B
        SN   = XOR1 ^ CI
        AND1 = A & B
        AND2 = XOR1 & CI
        OR1  = AND1 | AND2
        CON  = int(bool(OR1))

        with open(dot_file, "w") as f:
            f.write("digraph FA_boxed {\n  rankdir=LR;\n")
            f.write(f'  subgraph cluster_fa {{ label="{verilogFuncs.box_label(inst_name, "Full Adder")}"; '
                    "style=rounded;\n")
            f.write(f'    "XOR1" [label="XOR1 = {XOR1}", shape=box];\n')
            f.write(f'    "XOR2" [label="Sum  = {SN}",   shape=box];\n')
            f.write(f'    "AND1" [label="AND1 = {AND1}", shape=box];\n')
            f.write(f'    "AND2" [label="AND2 = {AND2}", shape=box];\n')
            f.write(f'    "OR1"  [label="OR1  = {OR1}",  shape=box];\n')
            f.write("  }\n")

            # I/O bubbles
            for sig, role, val in [(sigA, "A", A), (sigB, "B", B), (sigCI, "CI", CI)]:
                f.write(f'  "{sig}" [label="{sig}\\n({role}={val})", '
                        'shape=circle, style=filled, fillcolor=lightblue];\n')
            f.write(f'  "SN_OUT"  [label="SN  = {SN}",  shape=circle, style=filled, fillcolor=red];\n')
            f.write(f'  "CON_OUT" [label="CON = {CON}", shape=circle, style=filled, fillcolor=red];\n')

            # wiring
            f.write(f'  "{sigA}" -> "XOR1"; "{sigB}" -> "XOR1";\n')
            f.write(f'  "XOR1" -> "XOR2"; "{sigCI}" -> "XOR2";\n')
            f.write('  "XOR2" -> "SN_OUT";\n')
            f.write(f'  "{sigA}" -> "AND1"; "{sigB}" -> "AND1";\n')
            f.write(f'  "XOR1" -> "AND2"; "{sigCI}" -> "AND2";\n')
            f.write('  "AND1" -> "OR1"; "AND2" -> "OR1";\n')
            f.write('  "OR1"  -> "CON_OUT";\n}\n')

    # ------------------------------------------------------------------ #
    #  Half‑Adder boxed view
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_HA_dot(dot_file, inst_name, port_info):
        (sigA, A), (sigB, B) = port_info["A"], port_info["B"]
        SN  = A ^ B
        CON = A & B

        with open(dot_file, "w") as f:
            f.write("digraph HA_boxed {\n  rankdir=LR;\n")
            f.write(f'  subgraph cluster_ha {{ label="{verilogFuncs.box_label(inst_name, "Half Adder")}"; '
                    "style=rounded;\n")
            f.write(f'    "XOR" [label="Sum   = {SN}",  shape=box];\n')
            f.write(f'    "AND" [label="Carry = {CON}", shape=box];\n')
            f.write("  }\n")
            for sig, role, val in [(sigA, "A", A), (sigB, "B", B)]:
                f.write(f'  "{sig}" [label="{sig}\\n({role}={val})", '
                        'shape=circle, style=filled, fillcolor=lightblue];\n')
            f.write(f'  "SN_OUT"  [label="SN  = {SN}",  shape=circle, style=filled, fillcolor=red];\n')
            f.write(f'  "CON_OUT" [label="CON = {CON}", shape=circle, style=filled, fillcolor=red];\n')
            f.write(f'  "{sigA}" -> "XOR"; "{sigB}" -> "XOR";\n')
            f.write('  "XOR" -> "SN_OUT";\n')
            f.write(f'  "{sigA}" -> "AND"; "{sigB}" -> "AND";\n')
            f.write('  "AND" -> "CON_OUT";\n}\n')

    # ------------------------------------------------------------------ #
    #  Entry point: visualise ANY primitive instance
    # ------------------------------------------------------------------ #
    @staticmethod
    def simulate_gate_instance(mods, top_name, inst_name, sigs, dbg=False):
        top = mods.get(top_name)
        if not top:
            print("Top module not found."); return
        tgt = next((i for i in top["instances"]
                    if i["instance_name"] == inst_name), None)
        if not tgt:
            print(f"Instance '{inst_name}' not found in top."); return

        gtype = tgt["module"]
        if gtype not in verilogFuncs.primitive_ports:
            print(f"Gate type '{gtype}' not supported."); return

        # collect port‑info {role: (net, val)}
        prefix = top_name + "_"
        roles  = verilogFuncs.primitive_ports[gtype]["inputs"] + \
                 verilogFuncs.primitive_ports[gtype]["outputs"]
        pinfo  = {}
        for idx, c in enumerate(tgt["connections"]):
            role = c.get("port", roles[idx] if idx < len(roles) else None)
            pinfo[role] = (c["signal"], sigs.get(prefix + c["signal"], 0))

        dot = f"{inst_name}_{gtype}.dot"
        if gtype == "FA":
            verilogFuncs.generate_FA_dot(dot, inst_name, pinfo)
        elif gtype == "HA":
            verilogFuncs.generate_HA_dot(dot, inst_name, pinfo)
        else:
            verilogFuncs.generate_primitive_dot(dot, gtype, inst_name, pinfo)

        if dbg:
            print(f"[DOT] wrote {dot}")

    # ------------------------------------------------------------------ #
    #  DOT helper routines for hierarchy view
    # ------------------------------------------------------------------ #
    @staticmethod
    def collect_graph_data(mods, mod_name, scope, sigs, G, seen, top):
        def add(u, v, lbl):
            if (u, v, lbl) not in G["edge_set"]:
                G["edge_set"].add((u, v, lbl))
                G["edges"].append((u, v, lbl))

        mod = mods.get(mod_name)
        if mod is None:
            return

        # top‑level I/O bubbles
        if mod_name == top:
            for p in mod["inputs"]:
                G["nodes"][scope + "IN_" + p] = {"label": f"{p}={sigs.get(scope+p,'X')}",
                                                "type": "input"}
            for p in mod["outputs"]:
                G["nodes"][scope + "OUT_" + p] = {"label": f"{p}={sigs.get(scope+p,'X')}",
                                                 "type": "output"}

        # instance nodes
        for inst in mod["instances"]:
            pretty = f'{inst["instance_name"]} ({inst["module"]})'
            G["nodes"][scope + inst["instance_name"]] = {"label": pretty, "type": "module"}

        # edges into instances
        for inst in mod["instances"]:
            mtype, iname = inst["module"], inst["instance_name"]
            nid   = scope + iname
            ins   = mods.get(mtype, {}).get("inputs",
                   verilogFuncs.primitive_ports.get(mtype, {}).get("inputs", []))
            ports = mods.get(mtype, {}).get("ports", ins +
                    verilogFuncs.primitive_ports.get(mtype, {}).get("outputs", []))
            for idx, c in enumerate(inst["connections"]):
                role = c.get("port", ports[idx] if idx < len(ports) else None)
                sig  = c["signal"]
                if role in ins:
                    drv = verilogFuncs.find_driver_for_signal(
                        mods, mod, scope, sig, sigs, top, G)
                    add(drv, nid, f"{sig}={sigs.get(scope+sig,'X')}")

        # edges out of instances
        for inst in mod["instances"]:
            mtype, iname = inst["module"], inst["instance_name"]
            nid   = scope + iname
            outs  = mods.get(mtype, {}).get("outputs",
                    verilogFuncs.primitive_ports.get(mtype, {}).get("outputs", []))
            ports = mods.get(mtype, {}).get("ports", [])
            for idx, c in enumerate(inst["connections"]):
                role = c.get("port", ports[idx] if idx < len(ports) else None)
                sig  = c["signal"]
                if role in outs:
                    lds = verilogFuncs.find_loads_for_signal(
                        mods, mod, scope, sig, sigs, top, G)
                    for ld in lds:
                        add(nid, ld, f"{sig}={sigs.get(scope+sig,'X')}")

        # recurse into sub‑instances
        for inst in mod["instances"]:
            key = (inst["module"], scope + inst["instance_name"])
            if key not in seen:
                seen.add(key)
                verilogFuncs.collect_graph_data(
                    mods, inst["module"], scope + inst["instance_name"] + "_",
                    sigs, G, seen, top)

    @staticmethod
    def find_driver_for_signal(mods, parent, scope, sig, sigs, top, G):
        if parent["name"] == top and sig in parent["inputs"]:
            return scope + "IN_" + sig
        for inst in parent["instances"]:
            outs = mods.get(inst["module"], {}).get("outputs",
                   verilogFuncs.primitive_ports.get(inst["module"], {}).get("outputs", []))
            pts  = mods.get(inst["module"], {}).get("ports", outs)
            for idx, c in enumerate(inst["connections"]):
                role = c.get("port", pts[idx] if idx < len(pts) else None)
                if c["signal"] == sig and role in outs:
                    return scope + inst["instance_name"]
        dummy = scope + "DRV_" + sig
        if dummy not in G["nodes"]:
            G["nodes"][dummy] = {"label": f"DRV({sig})", "type": "module"}
        return dummy

    @staticmethod
    def find_loads_for_signal(mods, parent, scope, sig, sigs, top, G):
        lds = []
        if parent["name"] == top and sig in parent["outputs"]:
            lds.append(scope + "OUT_" + sig)
        for inst in parent["instances"]:
            ins = mods.get(inst["module"], {}).get("inputs",
                   verilogFuncs.primitive_ports.get(inst["module"], {}).get("inputs", []))
            pts = mods.get(inst["module"], {}).get("ports", ins)
            for idx, c in enumerate(inst["connections"]):
                role = c.get("port", pts[idx] if idx < len(pts) else None)
                if c["signal"] == sig and role in ins:
                    lds.append(scope + inst["instance_name"])
        return lds

    @staticmethod
    def compute_levels(G):
        nodes = list(G["nodes"])
        adj   = {n: [] for n in nodes}
        indeg = {n: 0 for n in nodes}
        for u, v, _ in G["edges"]:
            adj[u].append(v)
            indeg[v] += 1
        lvl, q = {}, deque([n for n in nodes if indeg[n] == 0])
        for n in q:
            lvl[n] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    lvl[v] = lvl[u] + 1
                    q.append(v)
        return lvl

    # ------------------------------------------------------------------ #
    #  Layered DOT writer for whole design
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_layered_dot(mods, top_module_name, dot_file_name, signal_values):
        """
        Generate a layered Graphviz DOT of the full hierarchy,
        filtering out dummy DRIVER nodes.
        """
        # 1) Collect full graph data
        G = {"nodes": {}, "edges": [], "edge_set": set()}
        verilogFuncs.collect_graph_data(
            mods,
            top_module_name,                # mod_name
            top_module_name + "_",          # scope
            signal_values,                  # sigs
            G,                              # graph_data
            set(),                          # seen set
            top_module_name                 # top
        )

        # 2) Remove dummy DRIVER nodes
        dummy = {
            n for n, attr in G["nodes"].items()
            if attr["type"] == "module" and attr["label"].startswith("DRV(")
        }
        for n in dummy:
            G["nodes"].pop(n, None)
        G["edges"] = [(u, v, lbl) for (u, v, lbl) in G["edges"]
                      if u not in dummy and v not in dummy]
        G["edge_set"] = set(G["edges"])

        # 3) Compute levels for ranking
        levels = verilogFuncs.compute_levels(G)
        ranks = {}
        for node, lvl in levels.items():
            ranks.setdefault(lvl, []).append(node)

        # 4) Emit DOT
        with open(dot_file_name, "w") as f:
            f.write("digraph hierarchy {\n  rankdir=LR;\n\n")

            # same‑rank clusters
            for lvl, nodes in sorted(ranks.items()):
                f.write("  { rank=same; " +
                        " ".join(f"\"{n}\";" for n in nodes) +
                        " }\n")
            f.write("\n")

            # nodes
            for n, attr in G["nodes"].items():
                shape = "box"
                extra = ""
                if attr["type"] == "input":
                    shape, extra = "circle", "style=filled fillcolor=lightblue"
                elif attr["type"] == "output":
                    shape, extra = "doublecircle", "style=filled fillcolor=red"
                f.write(f'  "{n}" [label="{attr["label"]}", shape={shape} {extra}];\n')

            f.write("\n")
            # edges
            for u, v, lbl in G["edges"]:
                f.write(f'  "{u}" -> "{v}" [label="{lbl}"];\n')
            f.write("}\n") 