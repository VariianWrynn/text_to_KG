import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pylab import mpl

def group_nodes_by_common_target(G, relationships):
    node_groups = {}
    for relationship in relationships:
        source, target = relationship.split(" -> ")
        source, target = source.strip(), target.strip()
        if source > target:
            source, target = target, source
        if target not in node_groups:
            node_groups[target] = []
        if source not in node_groups[target]:
            node_groups[target].append(source)
    return node_groups

def assign_colors_to_groups(node_groups):
    group_colors = {}
    for group, nodes in node_groups.items():
        most_common_node = max(nodes, key=lambda node: nodes.count(node))
        group_colors[group] = plt.cm.tab20(len(group_colors) % 20)
        group_colors[most_common_node] = group_colors[group]
    return group_colors

def plot_graph(graph, labels=None, title=None):
    data = pd.read_csv("output_similarity_score.csv")
    
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    relationships = data['combined']
    scores = data.iloc[:, 1:].mean(axis=1)

    relationships = relationships.str.replace(": contextual proximity", "")

    threshold = 0.13
    filtered_relationships = relationships[scores > threshold]
    filtered_scores = scores[scores > threshold]

    G = nx.Graph()

    for relationship, score in zip(filtered_relationships, filtered_scores):
        source, target = relationship.split(" -> ")
        source, target = source.strip(), target.strip()
        if source > target:
            source, target = target, source
        G.add_edge(source, target, weight=score)

    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    normalized_colors = [(weight - min_weight) / (max_weight - min_weight) for weight in edge_weights]

    colors = ["#FFFF00", "#FF0000"] 
    custom_cmap = LinearSegmentedColormap.from_list("CustomMap", colors)

    plt.figure(figsize=(20, 20))  # 增加图像大小
    pos = nx.spring_layout(G, seed = 42, k=0.3)  # 调整 k 值以增加节点间距

    node_groups = group_nodes_by_common_target(G, filtered_relationships)
    group_colors = assign_colors_to_groups(node_groups)

    node_color_list = []
    for node in G.nodes():
        for target, sources in node_groups.items():
            if node in sources or node == target:
                node_color_list.append(group_colors[target])
                break
        else:
            node_color_list.append("#A0CBE2")  # 默认颜色

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_color_list, alpha=0.9)
    edge_collection = nx.draw_networkx_edges(
        G, pos, edge_color=normalized_colors, edge_cmap=custom_cmap, width=2, arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0', min_source_margin=15, min_target_margin=15
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d:.2f}' for u, v, d in G.edges(data="weight")})

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array(edge_weights)
    plt.colorbar(sm, label="Edge Weight (Relevance)", orientation="vertical", pad=0.02, ax=plt.gca())

    plt.title("Optimized Knowledge Graph", fontsize=20)
    plt.savefig("optimized_knowledge_graph.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_graph("output_similarity_score.csv")
