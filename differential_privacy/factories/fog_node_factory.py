from differential_privacy.model.fog_node import FogNode


class FogNodeFactory:
    def __init__(self, fog_node_config: dict):
        self.device_count = fog_node_config['device_count']
        self.gradient_folder_name = fog_node_config['gradient_folder']

    def create_fog_node(self) -> FogNode:
        return FogNode(self.device_count, self.gradient_folder_name)
