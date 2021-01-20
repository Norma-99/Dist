import logging
from differential_privacy.dataset import Dataset
from .network_component import NetworkComponent


logger = logging.getLogger(__name__)


class Device(NetworkComponent):
    def __init__(self, dataset: Dataset):
        NetworkComponent.__init__(self)
        self.dataset: Dataset = dataset
        self.fog_node_address: int = None

    def set_fog_node(self, fog_node_address: int):
        self.fog_node_address = fog_node_address

    def on_data_receive(self, data):
        self.send({'dataset': self.dataset}, self.fog_node_address)
