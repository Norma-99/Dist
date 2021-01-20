import logging
from .network_component import NetworkComponent


logger = logging.getLogger(__name__)


class Network:
    def __init__(self):
        self._components = {}

    def add_component(self, component: NetworkComponent):
        component.set_network(self)
        self._components[component.get_address()] = component

    def send(self, data: dict):
        destination = data['destination']
        if destination not in self._components:
            logger.warning(
                    'Packet lost, destination %d not found',
                    destination)
            return
        destination_component = self._components[destination]
        destination_component.on_data_receive(data)
