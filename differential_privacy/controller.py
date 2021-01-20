import logging
import tensorflow as tf
from typing import List
from differential_privacy.dataset import Dataset
from differential_privacy.model import Server
from differential_privacy.factories.fog_node_factory import FogNodeFactory
from differential_privacy.factories.device_factory import DeviceFactory
from differential_privacy.model import *


logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config):
        self.network: Network = Network()
        server = self._create_server(config['server_config'], config['fog_node_count'])
        fog_nodes = self._create_fog_nodes(config, server)
        self.device_addresses = self._create_devices(config, fog_nodes)
        self.trigger = NetworkComponent()
        self.network.add_component(self.trigger)
        self.iterations = config['iterations']

    def _create_server(self, server_config: dict, fog_node_count: int) -> Server:
        validation_dataset = Dataset.from_file(server_config['dataset_path'])
        tf_model = None
        with open(server_config['model_path'], 'r') as json_file:
            tf_model = tf.keras.models.model_from_json(json_file.read())
        neural_network = NeuralNetwork(
                tf_model,
                server_config['epochs'],
                validation_dataset,
                server_config['trace_path'])
        server = Server(neural_network, fog_node_count)
        self.network.add_component(server)
        logger.info('Server created at %d', server.get_address())
        return server

    def _create_fog_nodes(self, config: dict, server: Server) -> List[FogNode]:
        fog_node_factory = FogNodeFactory(config['fog_node_config'])
        fog_nodes = [fog_node_factory.create_fog_node() for _ in range(config['fog_node_count'])]
        for fog_node in fog_nodes:
            fog_node.set_server(server.get_address())
            self.network.add_component(fog_node)
            logger.info('Fog node created at %d', fog_node.get_address())
        return fog_nodes

    def _create_devices(self, config: dict, fog_nodes: List[FogNode]) -> List[int]:
        device_factory = DeviceFactory(config['device_config'])
        total_device_count = config['fog_node_count'] * config['fog_node_config']['device_count']
        devices = device_factory.create_devices(total_device_count)
        for index, device in enumerate(devices):
            fog_node_index = index % len(fog_nodes)
            fog_node = fog_nodes[fog_node_index]
            device.set_fog_node(fog_node.get_address())
            self.network.add_component(device)
            logger.info('Device created at %d', device.get_address())
        return [device.get_address() for device in devices]

    def run(self):
        for iteration in range(self.iterations):
            for device_address in self.device_addresses:
                self.trigger.send({}, device_address)
