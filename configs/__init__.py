from .config_parser import Config
CONF = Config('configs/config.yaml').getConfig()

__all__ = ['CONF']