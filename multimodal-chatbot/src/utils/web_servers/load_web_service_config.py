import yaml
from pyprojroot import here


class LoadWebServicesConfig:

    def __init__(self) -> None:
        with open(here("configs/web_services.yml")) as cfg:
            web_service_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.llava_service_port = web_service_config["llava_service_port"]
        self.rag_reference_service_port = web_service_config["rag_reference_service_port"]
        self.stable_diffusion_service_port = web_service_config["stable_diffusion_service_port"]
        self.whisper_service_port = web_service_config["whisper_service"]
