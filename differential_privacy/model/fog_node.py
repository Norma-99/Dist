import logging
from typing import List
from .network_component import NetworkComponent
from .neural_network import NeuralNetwork
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient
from differential_privacy.factories.gradient_factory import GradientFactory

logger = logging.getLogger(__name__)


class FogNode(NetworkComponent):
    def __init__(self, device_count: int, gradient_folder_name):
        NetworkComponent.__init__(self)
        self._device_count: int = device_count
        self._gradient_folder_name = gradient_folder_name
        self._server_address: int = 0
        self._gradients: List[Gradient] = []
        self._generalization_datasets = {}
        self._current_device: int = 0
        self._current_dataset: Dataset = Dataset(None, None)
        self._current_device_counter: int = 0

    def on_data_receive(self, data: dict):
        logger.debug("Received %s", str(data))
        if 'dataset' in data:
            self._current_device = data['origin']
            self._process_dataset(data)
        elif 'neural_network' in data:
            self._train_network(data['neural_network'].clone())
            if self._has_all_gradients():
                self.on_iteration_end(data['neural_network'].clone())

    def _save_generalisation_fragment(self):
        self._current_device_counter = self._current_device_counter + 1
        self._generalization_datasets[self._current_device_counter] = self._current_dataset.get_generalisation_fragment(1)

    def _process_dataset(self, data: dict):
        self._current_dataset = data['dataset']
        logger.info('Processed dataset from %d', self._current_device_counter)
        self._save_generalisation_fragment()
        self.send({}, self.server_address)

    def _train_network(self, neural_network: NeuralNetwork):
        self._gradients.append(neural_network.fit(self._current_dataset))
        neural_network.save_trace(self._current_device)

    def set_server(self, server_address: int):
        self.server_address = server_address

    def on_iteration_end(self, neural_network: NeuralNetwork):
        gradient_folder = GradientFactory.from_name(
            self._gradient_folder_name,
            dataset=Dataset.join(self._generalization_datasets),
            neural_network=neural_network)
        gradient = gradient_folder.fold(self._gradients)
        self._gradients.clear()
        self._current_device_counter = 0
        self.send({'gradient': gradient}, self.server_address)

    def _has_all_gradients(self):
        return self._current_device_counter == self._device_count
