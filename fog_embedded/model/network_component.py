import logging


logger = logging.getLogger(__name__)


class NetworkComponent:
    _last_used_address = 0

    def __init__(self):
        self._address = self._get_unique_address()
        self._network = None

    def get_address(self):
        return self._address

    def set_network(self, network):
        self._network = network

    def send(self, data: dict, to: int):
        if self._network is None:
            raise ValueError('Network not set')
        data['origin'] = self._address
        data['destination'] = to
        self._network.send(data)

    def on_data_receive(self, data: dict):
        logger.info('Received %s', str(data))

    @staticmethod
    def _get_unique_address():
        NetworkComponent._last_used_address += 1
        return NetworkComponent._last_used_address
