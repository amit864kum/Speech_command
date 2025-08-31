class P2PNetwork:
    def __init__(self):
        # A dictionary to represent the network, where keys are node IDs
        # and values are the node objects themselves.
        self.nodes = {}

    def add_node(self, node):
        """
        Adds a new node to the network.
        """
        self.nodes[node.node_id] = node
        
    def get_node(self, node_id):
        """
        Retrieves a node object from the network.
        """
        return self.nodes.get(node_id)
        
    def broadcast_model(self, model_state_dict):
        """
        Simulates broadcasting a model to all nodes in the network.
        """
        for node in self.nodes.values():
            node.receive_global_model(model_state_dict)
            
    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.values())