# Expression graph visualization
from typing import Union
from graphviz import Digraph
from IPython.display import display

from utils.engine import Value

def trace(root:Value)->Union[set[Value],set[tuple[Value, Value]]]:
    nodes, edges = set(), set()
    def build(v:Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((v,child))
                build(child)
    build(root)
    return nodes, edges

def draw_graph(root:Value):
    graph = Digraph(name="Expression graph", format = "svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    op_id_dict, node_id_dict = dict(), dict()
    for node in nodes:
        node_id = str(id(node))
        node_id_dict[node] = node_id
        graph.node(node_id, label = f"{node.label}|data={node.data:0.4f}| grad={node.grad:0.4f}", shape = "record")
        if node._op:
            #create op node
            op_node_id = node_id + "_op"
            op_id_dict[node] = op_node_id
            graph.node(op_node_id, label = node._op)
            graph.edge(op_node_id, node_id, label = "output")
    #Now connect the children to the operation nodes
    for parent,child in edges:
        if parent._op:
            graph.edge(node_id_dict[child], op_id_dict[parent], label = "input")
    return graph


def display_graph(root:Value):
    graph = draw_graph(root)
    return display(graph)


