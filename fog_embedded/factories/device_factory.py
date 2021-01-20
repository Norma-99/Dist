from typing import List
from differential_privacy.model.device import Device
from differential_privacy.dataset import Dataset


class DeviceFactory:
    def __init__(self, device_config: dict):
        dataset_path = device_config['splittable_dataset']
        self.dataset = Dataset.from_file(dataset_path)

    def create_devices(self, device_count) -> List[Device]:
        devices = []
        for i in range(device_count):
            devices.append(Device(self.dataset.get_split(i, device_count)))
        return devices
