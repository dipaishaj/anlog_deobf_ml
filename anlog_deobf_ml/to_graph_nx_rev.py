import sys
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import graph_anadec


class CreateGraph:
    def __init__(self, vertices, wire, features):
        self.vertices = vertices
        self.wire = wire
        self.features = features
        self.index_list = dict()
        self.comp_index = dict()
        self.dev_cnt = 0
        self.comp_cnt = 0

        return

    def draw_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        pairs = [(key, value)
                 for key, values in self.wire.items()
                 for value in values]
        print("Edges in Graph: ")
        # for pair in pairs:
            # print(pair)

        G.add_edges_from(pairs)
        size = G.number_of_nodes()

        print("number_of_nodes: ", G.number_of_nodes())
        print("number_of_edges: ", G.number_of_edges())

        color_map = []
        for node in G:
            if node.startswith('r'):
                color_map.append('cyan')
            elif node.startswith('c'):
                color_map.append('green')
            elif node.startswith('l'):
                color_map.append('blue')
            elif node.startswith('p'):
                color_map.append('yellow')
            elif node.startswith('n'):
                color_map.append('magenta')
            elif node.startswith('x'):
                color_map.append('gray')
            elif node.startswith('vin'):
                color_map.append('gray')
            elif node.startswith('y'):
                color_map.append('red')
            elif node.startswith('vout'):
                color_map.append('red')
            else:
                color_map.append('pink')
        nx.draw(G, node_color=color_map, with_labels=True)

        plt.savefig("Mygraph_1.png")
        plt.show()

        return

    def nm2index(self, nm):
        if nm in self.index_list:
            return self.index_list[nm]
        else:
            cnt = self.dev_cnt
            self.index_list[nm] = cnt
            self.dev_cnt += 1
            return cnt

    def comp2index(self, comp):
        if comp in self.comp_index:
            return self.comp_index[comp]
        else:
            cnt = self.comp_cnt
            self.comp_index[comp] = cnt
            self.comp_cnt += 1
            return cnt

#
# def main(b_file: str):
#     print("Parsing the file: ", b_file)
#     file_obj = graph_anadec.AnaCircuit()
#     vert, wires, feat, ckt_name, circuit_X, circuit_Y = file_obj.read_bench(b_file)
#     # print("Parsed bench file for : ", ckt_name)
#     gh = CreateGraph(vert, wires, feat)
#     gh.draw_graph()
#     return
#
#
# if __name__ == "__main__":
#     if len(sys.argv) == 2:
#         ben_file = sys.argv[1]
#         main(ben_file)
#     else:
#         print("No arguments passed: pass the bench file ")
