import graphviz
import tensorrt as trt


def trt_network_to_dot_graph(network):
    dot = graphviz.Digraph(comment="Network")

    # add nodes (layers)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        dot.node(layer.name)

    # add nodes (inputs)
    for i in range(network.num_inputs):
        dot.node(network.get_input(i).name)

    # add nodes (outputs)
    for i in range(network.num_outputs):
        dot.node(network.get_output(i).name)

    # add layer->layer edges
    # Use a seen set to deduplicate edges, in case the same tensor is consumed
    # by multiple input slots of the same destination layer (e.g. elementwise ops).
    seen_edges = set()
    for a in range(network.num_layers):
        layer_a = network.get_layer(a)

        for i in range(layer_a.num_outputs):
            output_i = layer_a.get_output(i)

            for b in range(network.num_layers):
                layer_b = network.get_layer(b)

                for j in range(layer_b.num_inputs):
                    input_j = layer_b.get_input(j)

                    if output_i == input_j:
                        edge_key = (layer_a.name, layer_b.name, str(input_j.shape))
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            dot.edge(layer_a.name, layer_b.name, label=str(input_j.shape))

    # add input->layer edges
    seen_input_edges = set()
    for i in range(network.num_inputs):
        input_i = network.get_input(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_inputs):
                input_j = layer_b.get_input(j)

                if input_i == input_j:
                    edge_key = (input_i.name, layer_b.name, str(input_j.shape))
                    if edge_key not in seen_input_edges:
                        seen_input_edges.add(edge_key)
                        dot.edge(input_i.name, layer_b.name, label=str(input_j.shape))

    # add layer->output edges
    seen_output_edges = set()
    for i in range(network.num_outputs):
        input_i = network.get_output(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_outputs):
                input_j = layer_b.get_output(j)

                if input_i == input_j:
                    edge_key = (layer_b.name, input_i.name, str(input_j.shape))
                    if edge_key not in seen_output_edges:
                        seen_output_edges.add(edge_key)
                        dot.edge(layer_b.name, input_i.name, label=str(input_j.shape))

    return dot
