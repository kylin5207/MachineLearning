import networkx as nx
import collections
import matplotlib.pyplot as plt
import queue

# class BinaryTreePlotter:
#     def __init__(self, root):
#         self.root = root
#
#     def create_graph(self, G, node, pos={}, x=0, y=0, layer=1):
#         pos[node.id] = (x, y)
#         if node.left_nodeid:
#             G.add_edge(node.id, node.left_nodeid.id, label=f"{node.sitename}")
#             l_x, l_y = x - 1 / 2 ** layer, y - 1
#             l_layer = layer + 1
#             self.create_graph(G, node.left_nodeid, x=l_x, y=l_y, pos=pos, layer=l_layer)
#         if node.right_nodeid:
#             G.add_edge(node.id, node.right_nodeid.id, label=f"{node.sitename}")
#             r_x, r_y = x + 1 / 2 ** layer, y - 1
#             r_layer = layer + 1
#             self.create_graph(G, node.right_nodeid, x=r_x, y=r_y, pos=pos, layer=r_layer)
#         return (G, pos)
#
#     def draw(self):
#         # 创建有向图
#         graph = nx.DiGraph()
#         graph, pos = self.create_graph(graph, self.root)
#         fig, ax = plt.subplots(figsize=(20, 10))  # 比例可以根据树的深度适当调节
#         nx.draw_networkx(graph, pos, ax=ax, node_size=500)
#         plt.show()

class BinaryTreePlotter:
    def __init__(self, root):
        self.root = root

    def create_graph(self, G, node, pos={}, x=0, y=0, layer=1):
        pos[node.id] = (x, y)

        if node.left_nodeid:
            if node.left_nodeid.type == "subnode":
                G.add_node(node.left_nodeid.id, label=f"P: {node.left_nodeid.sitename}, R: {node.left_nodeid.split_rank}, a={node.left_nodeid.available_nodes}")
            else:
                G.add_node(node.left_nodeid.id, label=f"Leaf")
            G.add_edge(node.id, node.left_nodeid.id)
            l_x, l_y = x - 1 / 2 ** layer, y - 1
            # l_x, l_y = x - layer / 2, y - 1
            l_layer = layer + 1
            self.create_graph(G, node.left_nodeid, x=l_x, y=l_y, pos=pos, layer=l_layer)
        if node.right_nodeid:
            if node.right_nodeid.type == "subnode":
                G.add_node(node.right_nodeid.id, label=f"P: {node.right_nodeid.sitename}, R: {node.right_nodeid.split_rank}, a={node.right_nodeid.available_nodes}")
            else:
                G.add_node(node.right_nodeid.id, label=f"Leaf")
            G.add_edge(node.id, node.right_nodeid.id)
            r_x, r_y = x + 1 / 2 ** layer, y - 1
            # r_x, r_y = x + layer / 2, y - 1
            r_layer = layer + 1
            self.create_graph(G, node.right_nodeid, x=r_x, y=r_y, pos=pos, layer=r_layer)
        return (G, pos)

    def draw(self, node_size=1000):
        # 创建有向图
        fig, ax = plt.subplots(figsize=(30, 15))  # 比例可以根据树的深度适当调节
        graph = nx.DiGraph()

        # edge_labels = nx.get_edge_attributes(graph, 'name')  # 获取边的name属性
        graph.add_node(self.root.id, label=f"P: {self.root.sitename}, R: {self.root.split_rank}, a={self.root.available_nodes}")
        graph, pos = self.create_graph(graph, self.root)

        node_labels = nx.get_node_attributes(graph, 'label')  # 获取节点的desc属性
        node_color = []
        for node in node_labels:
            label = node_labels[node]
            if label == "Leaf":
                color = "green"
            else:
                color = "red"
            node_color.append(color)

        nx.draw(graph, pos, edge_color="grey", node_size=node_size)  # 画图，设置节点大小

        nx.draw_networkx_nodes(graph, pos, label=node_labels, node_color=node_color, node_size=node_size)

        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=20)  # 将label属性，显示在节点上
        # nx.draw_networkx(graph, pos, with_labels=False, ax=ax, node_size=1000)
        # nx.draw_networkx_edge_labels(graph, pos, font_size=20)  # 将name属性，显示在边上
        plt.show()

