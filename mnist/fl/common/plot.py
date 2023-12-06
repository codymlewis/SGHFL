import graphviz

def plot_network(network_arch, filename="network_architecture", view=False):
    dot = graphviz.Graph(filename=filename, format="pdf")
    server_node(dot, network_arch, "server")
    client_count = 0
    if isinstance(network_arch['clients'], int):
        for i in range(network_arch['clients']):
            dot.node(f"{i}", f"Client {i}")
            dot.edge("server", f"{i}")
    elif isinstance(network_arch['clients'], dict):
        ms_name = "middle server"
        server_node(dot, network_arch['clients'], ms_name)
        dot.edge("server", ms_name)
        for i in range(network_arch['clients']['clients']):
            dot.node(f"{i}", f"Client {i}")
            dot.edge(ms_name, f"{i}")
    else:
        for i, subnet in enumerate(network_arch['clients']):
            ms_name = f"middle server {i}"
            server_node(dot, subnet, ms_name)
            dot.edge("server", ms_name)
            plot_subnet(dot, ms_name, subnet)
    dot.render(view=view)


def plot_subnet(dot, parent_name, subnet):
    if isinstance(subnet['clients'], int):
        for i in range(subnet['clients']):
            dot.node(f"{parent_name} {i}", f"Client {i}")
            dot.edge(parent_name, f"{parent_name} {i}")
    elif isinstance(subnet['clients'], dict):
        ms_name = f"{parent_name} middle server"
        server_node(dot, subnet['clients'], ms_name)
        dot.edge(parent_name, ms_name)
        for i in range(subnet['clients']['clients']):
            dot.node(f"{parent_name} {i}", f"Client {i}")
            dot.edge(ms_name, f"{parent_name} {i}")
    else:
        for i, subnet in enumerate(subnet['clients']):
            ms_name = f"{parent_name} middle server {i}"
            server_node(dot, subnet, ms_name)
            dot.edge(parent_name, ms_name)
            plot_subnet(dot, ms_name, subnet)


def server_node(dot, network_arch, kwname):
    label = kwname.title()
    for k, v in network_arch.items():
        if k != "clients":
            label += f"\n{k.title()}: {str(v)}"
    dot.attr("node", shape="box")
    dot.node(kwname, label)
    dot.attr("node", shape="ellipse")
